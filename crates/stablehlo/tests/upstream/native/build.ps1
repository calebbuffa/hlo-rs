[CmdletBinding()]
param(
    [string]$ReferenceDir,
    [string]$VcVars64 = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat',
    [ValidateRange(1, 16)]
    [int]$Jobs = 3
)

$ErrorActionPreference = 'Stop'
$crateRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$workspaceRoot = (Resolve-Path (Join-Path $crateRoot '..\..')).Path
if (!$ReferenceDir) {
    $ReferenceDir = Join-Path $workspaceRoot 'extern\stablehlo'
}
$ReferenceDir = (Resolve-Path $ReferenceDir).Path
$referenceRevision = & git --no-pager -C $ReferenceDir rev-parse HEAD
if ($LASTEXITCODE -ne 0) {
    throw "Cannot read StableHLO reference revision: $ReferenceDir"
}
$expectedReferenceRevision = '25605c9f48a9aec305876d60d26b028398d298d2'
if ($referenceRevision.Trim() -ne $expectedReferenceRevision) {
    throw "StableHLO reference must be at $expectedReferenceRevision; align extern\stablehlo explicitly before building."
}
if (!(Test-Path $VcVars64 -PathType Leaf)) {
    throw "Visual Studio C++ environment script not found: $VcVars64"
}
$revision = '2b35fb9283a44aae92cb72a41d077ee8de2be7ad'
$expectedHash = 'B61C85C97CDE1C3E96EA7CFD09737C24578D6D0656C392D79CE402DA6904B828'
$root = Join-Path $workspaceRoot 'target\reference-1.22'
$archive = Join-Path $root 'llvm-2b35fb9.tar.gz'
$source = Join-Path $workspaceRoot "extern\llvm-project-$revision"
$build = Join-Path $root 'build'
New-Item -ItemType Directory -Force $root, $source | Out-Null
if (!(Test-Path $archive -PathType Leaf)) {
    & curl.exe --fail --location --output "$archive.download" `
        "https://codeload.github.com/llvm/llvm-project/tar.gz/$revision"
    if ($LASTEXITCODE -ne 0) {
        throw "LLVM source download failed: exit $LASTEXITCODE"
    }
    Move-Item "$archive.download" $archive
}
if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne $expectedHash) {
    throw "LLVM source archive does not match the pinned SHA256: $archive"
}
$stamp = Join-Path $source 'extracted-archive.sha256'
if (!(Test-Path $stamp) -or (Get-Content $stamp -Raw).Trim() -ne $expectedHash) {
    $members = 'llvm', 'mlir', 'cmake', 'third-party', 'libc' |
        ForEach-Object { "llvm-project-$revision/$_" }
    & tar.exe -xzf $archive -C $source --strip-components=1 @members
    if ($LASTEXITCODE -ne 0) {
        throw "LLVM source extraction failed: exit $LASTEXITCODE"
    }
    Set-Content $stamp $expectedHash
}

$options = @(
    '-G Ninja'
    "-S `"$source\llvm`""
    "-B `"$build`""
    '-DCMAKE_BUILD_TYPE=Release'
    '-DCMAKE_C_COMPILER=clang-cl'
    '-DCMAKE_CXX_COMPILER=clang-cl'
    '-DLLVM_ENABLE_PROJECTS=mlir'
    '-DLLVM_EXTERNAL_PROJECTS="stablehlo;rustvalidator"'
    "-DLLVM_EXTERNAL_STABLEHLO_SOURCE_DIR=`"$ReferenceDir`""
    "-DLLVM_EXTERNAL_RUSTVALIDATOR_SOURCE_DIR=`"$PSScriptRoot`""
    "-DMLIR_INCLUDE_DIRS=`"$source\mlir\include`""
    '-DLLVM_TARGETS_TO_BUILD=X86'
    '-DLLVM_BUILD_TOOLS=OFF'
    '-DLLVM_INCLUDE_TESTS=OFF'
    '-DLLVM_INCLUDE_EXAMPLES=OFF'
    '-DLLVM_INCLUDE_BENCHMARKS=OFF'
    '-DLLVM_ENABLE_ASSERTIONS=ON'
    '-DLLVM_ENABLE_BINDINGS=OFF'
    '-DLLVM_ENABLE_ZLIB=OFF'
    '-DLLVM_ENABLE_ZSTD=OFF'
    '-DLLVM_ENABLE_LIBXML2=OFF'
    '-DMLIR_INCLUDE_TESTS=OFF'
    '-DMLIR_INCLUDE_EXAMPLES=OFF'
    '-DMLIR_ENABLE_BINDINGS_PYTHON=OFF'
    '-DLLVM_PARALLEL_LINK_JOBS=1'
)
$configure = 'cmake ' + ($options -join ' ')
$compile = "cmake --build `"$build`" --target stablehlo-rust-validator -j $Jobs"
& $env:ComSpec /d /c "call `"$VcVars64`" >nul && $configure && $compile"
if ($LASTEXITCODE -ne 0) {
    throw "Native validator build failed: exit $LASTEXITCODE"
}
$executable = Join-Path $build 'bin\stablehlo-rust-validator.exe'
& $executable --version
if ($LASTEXITCODE -ne 0) {
    throw "Native validator version check failed: exit $LASTEXITCODE"
}
Write-Output "Set STABLEHLO_UPSTREAM_NATIVE to $executable"
