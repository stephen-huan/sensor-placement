{ lib
, fetchPypi
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
, pytestCheckHook
, jax
, jaxlib
, torch
}:

buildPythonPackage rec {
  pname = "cola-ml";
  version = "0.0.5";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-r92S/PKijD0ozFU4sAWXg/mU1YphN37SmFywylIoE9o=";
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
  ];

  disabledTestPaths = [
    # tests that take a long time
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
