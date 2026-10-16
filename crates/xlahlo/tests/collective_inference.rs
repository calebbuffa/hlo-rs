use xlahlo::{Dimension as D, PrimitiveType as P, Shape, inference as i};

fn array(bounds: Vec<D>) -> Shape {
    Shape::dynamic_array(P::F32, bounds)
        .unwrap()
        .with_default_layout()
        .unwrap()
}

#[test]
fn gather_and_scatter_preserve_layout_but_follow_distinct_dynamic_axis_rules() {
    let input = array(vec![D::Bounded(6), D::Bounded(2)]);
    let gathered = i::infer_all_gather(std::slice::from_ref(&input), 0, 3).unwrap();
    assert_eq!(gathered.dimension(0).unwrap(), D::Static(18));
    assert_eq!(gathered.dimension(1).unwrap(), D::Bounded(2));
    assert_eq!(gathered.layout(), input.layout());
    let scattered = i::infer_reduce_scatter(std::slice::from_ref(&input), 0, 3).unwrap();
    assert_eq!(scattered.dimension(0).unwrap(), D::Unbounded);
    assert_eq!(scattered.dimension(1).unwrap(), D::Bounded(2));
    let unbounded = array(vec![D::Unbounded]);
    assert_eq!(
        i::infer_all_gather(std::slice::from_ref(&unbounded), 0, 2).unwrap(),
        unbounded
    );
    assert!(i::infer_reduce_scatter(&[input], 0, 4).is_err());
    assert!(i::infer_all_gather(&[Shape::array(P::PRED, vec![i64::MAX]).unwrap()], 0, 2).is_err());
    assert!(i::infer_all_gather(&[Shape::token()], 0, 1).is_err());
}

#[test]
fn variadic_context_shapes_and_empty_inference_match_source() {
    let a = array(vec![D::Static(2)]);
    let b = Shape::scalar(P::S32).unwrap();
    let gathered = i::infer_all_gather_start(std::slice::from_ref(&a), 0, 2).unwrap();
    assert_eq!(gathered.subshape(&[0]).unwrap(), a);
    assert_eq!(
        i::infer_all_gather_done(&gathered).unwrap().dimensions(),
        &[4]
    );
    let reduced = i::infer_all_reduce_start(&[a.clone(), b.clone()]).unwrap();
    assert_eq!(reduced, Shape::tuple(vec![a, b]).unwrap());
    assert_eq!(i::infer_all_reduce_done(&reduced).unwrap(), reduced);
    assert_eq!(
        i::infer_all_gather(&[], 0, 2).unwrap(),
        Shape::tuple(vec![]).unwrap()
    );
    assert_eq!(
        i::infer_all_reduce(&[]).unwrap(),
        Shape::tuple(vec![]).unwrap()
    );
    assert!(i::infer_all_gather_done(&Shape::token()).is_err());
}

#[test]
fn all_to_all_axis_exchange_checks_divisibility_dynamics_and_overflow() {
    let input = array(vec![D::Static(6), D::Static(4)]);
    assert_eq!(
        i::infer_all_to_all(&input, 0, 1, 3).unwrap().dimensions(),
        &[2, 12]
    );
    assert_eq!(i::infer_all_to_all(&input, 0, 0, 3).unwrap(), input);
    assert!(i::infer_all_to_all(&input, 0, 1, 4).is_err());
    assert!(i::infer_all_to_all(&input, -1, 1, 3).is_err());
    assert!(i::infer_all_to_all(&array(vec![D::Bounded(6)]), 0, 0, 3).is_err());
    assert_eq!(
        i::infer_all_to_all(&array(vec![D::Unbounded, D::Static(2)]), 0, 1, 3)
            .unwrap()
            .dimensions(),
        &[i64::MIN, 6]
    );
    assert!(i::infer_all_to_all(&array(vec![D::Static(0), D::Static(i64::MAX)]), 0, 1, 2).is_err());
    let mut wire = input.clone().into_proto();
    wire.layout.as_mut().unwrap().minor_to_major = vec![0, 1];
    let transposed = Shape::from_proto(wire).unwrap();
    assert_ne!(
        i::infer_all_to_all(&transposed, 0, 0, 3).unwrap().layout(),
        transposed.layout()
    );
}

#[test]
fn tuple_all_to_all_compares_full_shapes_except_memory_and_retains_each_layout() {
    let first = array(vec![D::Bounded(3), D::Static(2)]);
    let mut wire = first.clone().into_proto();
    wire.layout.as_mut().unwrap().memory_space = 5;
    let second = Shape::from_proto(wire.clone()).unwrap();
    assert_eq!(
        i::infer_all_to_all_tuple(&[first.clone(), second.clone()]).unwrap(),
        Shape::tuple(vec![first.clone(), second]).unwrap()
    );
    wire.layout.as_mut().unwrap().minor_to_major.reverse();
    assert!(i::infer_all_to_all_tuple(&[first, Shape::from_proto(wire).unwrap()]).is_err());
    assert!(i::infer_all_to_all_tuple(&[array(vec![D::Unbounded])]).is_err());
    assert!(i::infer_all_to_all_tuple(&[]).is_err());
    let nested = Shape::tuple(vec![Shape::token(), array(vec![D::Static(2)])]).unwrap();
    assert_eq!(
        i::infer_all_to_all_tuple(std::slice::from_ref(&nested)).unwrap(),
        Shape::tuple(vec![nested]).unwrap()
    );
}

#[test]
fn dynamic_root_broadcast_and_reduce_have_different_type_and_layout_constraints() {
    let input = array(vec![D::Static(2)]);
    let root = Shape::array(P::S32, vec![1])
        .unwrap()
        .with_default_layout()
        .unwrap();
    assert_eq!(
        i::infer_collective_broadcast(&[input.clone(), root.clone()], true).unwrap(),
        input
    );
    assert_eq!(
        i::infer_collective_reduce(&[input.clone(), root.clone()], true).unwrap(),
        input
    );
    let mut wire = root.into_proto();
    wire.layout.as_mut().unwrap().memory_space = 5;
    let root = Shape::from_proto(wire).unwrap();
    i::infer_collective_broadcast(&[input.clone(), root.clone()], true).unwrap();
    assert!(i::infer_collective_reduce(&[input.clone(), root.clone()], true).is_err());
    let nested = Shape::tuple(vec![input.clone()]).unwrap();
    assert_eq!(
        i::infer_collective_broadcast(&[nested.clone(), root.clone()], true).unwrap(),
        nested
    );
    assert!(i::infer_collective_reduce(&[nested, root], true).is_err());
    assert_eq!(
        i::infer_collective_broadcast(&[input.clone(), Shape::token()], false).unwrap(),
        input
    );
    assert!(i::infer_collective_broadcast(&[], true).is_err());
    assert_eq!(
        i::infer_ragged_all_to_all(&[Shape::token(), input.clone()]).unwrap(),
        input
    );
}

#[test]
fn static_gather_scatter_small_geometry_is_exact() {
    for size in 1..=8 {
        for shards in 1..=4 {
            let input = array(vec![D::Static(size)]);
            let gathered = i::infer_all_gather(std::slice::from_ref(&input), 0, shards).unwrap();
            assert_eq!(gathered.dimensions(), &[size * shards]);
            assert_eq!(
                i::infer_reduce_scatter(&[gathered], 0, shards).unwrap(),
                input
            );
            assert_eq!(
                i::infer_reduce_scatter(&[input], 0, shards).is_ok(),
                size % shards == 0
            );
        }
    }
}
