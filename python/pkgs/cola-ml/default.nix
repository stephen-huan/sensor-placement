{ lib
, fetchFromGitHub
, buildPythonPackage
, setuptools
, setuptools-scm
, scipy
, tqdm
, cola-plum-dispatch
, optree
, pytreeclass
, beartype
, typing-extensions
, pytest
, pytestCheckHook
, jax
, jaxlib
, torch
}:

buildPythonPackage {
  pname = "cola-ml";
  version = "0.0.5";

  src = fetchFromGitHub {
    owner = "wilson-labs";
    repo = "cola";
    rev = "4af199e4e7bd647241ecb60c3befea401b6dc7b2";
    hash = "sha256-MkswIJcXVWRFjSGXK0DGLZiM2JbDOqUrm2YMyis6q/Q=";
  };

  format = "pyproject";

  nativeBuildInputs = [
    setuptools
    setuptools-scm
  ];

  propagatedBuildInputs = [
    scipy
    tqdm
    cola-plum-dispatch
    optree
    pytreeclass
    beartype
    typing-extensions
    pytest
  ];

  nativeCheckInputs = [
    pytestCheckHook
    jax
    jaxlib
    torch
  ];

  disabledTests = [
    "test_get_lu_from_tridiagonal"
    "test_vmappable_constructor"
    "test_arnoldi_vjp"
    "test_lanczos_vjp"
    "test_unary"
    "test_arnoldi_matrix_market"
    "test_lanczos_matrix_market"
  ];

  disabledTestPaths = [
    # tests that take a long time
    "tests/linalg/test_inverse.py"
    "tests/linalg/test_logdet.py"
  ];

  pythonImportsCheck = [
    "cola"
  ];

  meta = with lib; {
    description = "Compositional Linear Algebra";
    homepage = "https://cola.readthedocs.io/en/latest/";
    license = licenses.asl20;
    maintainers = with maintainers; [ stephen-huan ];
  };
}
