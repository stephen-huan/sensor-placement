{ lib
, fetchFromGitHub
, buildPythonPackage
, poetry-core
, optax
, jaxopt
, jaxtyping
, tqdm
, simple-pytree
, tensorflow-probability
, beartype
, jax
, jaxlib
, orbax-checkpoint
, cola-ml
, pytestCheckHook
, networkx
, flax
}:

buildPythonPackage rec {
  pname = "gpjax";
  version = "0.8.0";

  # PyPi source doesn't contain tests
  src = fetchFromGitHub {
    owner = "JaxGaussianProcesses";
    repo = "GPJax";
    rev = "v${version}";
    sha256 = "sha256-s4ERd8qf2RsGX/EYfNxRklZKx9JtTKwXzQipAtIkHC8=";
  };

  format = "pyproject";

  patches = [
    # https://github.com/JaxGaussianProcesses/GPJax/pull/437
    ./test_gps.patch
  ];

  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace \
        'tensorflow-probability = "^0.19.0"' \
        'tensorflow-probability = ">=0.19.0"'

    substituteInPlace gpjax/fit.py \
      --replace \
        '_check_prng_key(key)' \
        '_check_prng_key("fit", key)'
  '';

  nativeBuildInputs = [
    poetry-core
    jaxlib
  ];

  propagatedBuildInputs = [
    optax
    jaxopt
    jaxtyping
    tqdm
    simple-pytree
    tensorflow-probability
    beartype
    jax
    orbax-checkpoint
    cola-ml
  ];

  nativeCheckInputs = [
    pytestCheckHook
    networkx
    flax
  ];

  disabledTestPaths = [
    # requires mktestdocs
    "tests/test_markdown.py"
  ];

  pythonImportsCheck = [
    "gpjax"
  ];

  meta = with lib; {
    description = "Gaussian processes in JAX";
    homepage = "https://docs.jaxgaussianprocesses.com";
    license = licenses.asl20;
    maintainers = with maintainers; [ stephen-huan ];
  };
}
