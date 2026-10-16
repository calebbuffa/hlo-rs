#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

#include "Eigen/Core"
#include "ml_dtypes/include/float8.h"
#include "ml_dtypes/include/mxfloat.h"

template <typename T>
void Convert(uint64_t source) {
  using Bits = std::conditional_t<
      sizeof(T) == 1, uint8_t,
      std::conditional_t<sizeof(T) == 2, uint16_t,
                         std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;
  const T value = static_cast<T>(std::bit_cast<double>(source));
  std::cout << std::hex << static_cast<uint64_t>(std::bit_cast<Bits>(value))
            << " " << std::isfinite(static_cast<double>(value)) << "\n";
}

int main() {
  static_assert(std::endian::native == std::endian::little);
  std::string type;
  uint64_t source;
  while (std::cin >> type) {
    if (!(std::cin >> std::hex >> source)) {
      std::cerr << "Invalid double bit pattern\n";
      return 2;
    }
    if (type == "f16") Convert<Eigen::half>(source);
    else if (type == "bf16") Convert<Eigen::bfloat16>(source);
    else if (type == "f32") Convert<float>(source);
    else if (type == "f64") Convert<double>(source);
    else if (type == "f8e3m4") Convert<ml_dtypes::float8_e3m4>(source);
    else if (type == "f8e4m3") Convert<ml_dtypes::float8_e4m3>(source);
    else if (type == "f8e5m2") Convert<ml_dtypes::float8_e5m2>(source);
    else if (type == "f8e4m3fn") Convert<ml_dtypes::float8_e4m3fn>(source);
    else if (type == "f8e4m3fnuz") Convert<ml_dtypes::float8_e4m3fnuz>(source);
    else if (type == "f8e4m3b11fnuz") Convert<ml_dtypes::float8_e4m3b11fnuz>(source);
    else if (type == "f8e5m2fnuz") Convert<ml_dtypes::float8_e5m2fnuz>(source);
    else if (type == "f8e8m0fnu") Convert<ml_dtypes::float8_e8m0fnu>(source);
    else if (type == "f4e2m1fn") Convert<ml_dtypes::float4_e2m1fn>(source);
    else if (type == "f6e2m3fn") Convert<ml_dtypes::float6_e2m3fn>(source);
    else if (type == "f6e3m2fn") Convert<ml_dtypes::float6_e3m2fn>(source);
    else {
      std::cerr << "Unknown primitive: " << type << "\n";
      return 2;
    }
  }
  return std::cin.bad() ? 2 : 0;
}
