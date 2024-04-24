{ lib
, fetchPypi
, buildPythonPackage
, setuptools
, jax
, jaxlib
, typing-extensions
, pytestCheckHook
, pytest-benchmark
, optax
}:

buildPythonPackage rec {
  pname = "pytreeclass";
  version = "0.9.2";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-28KxPOrUxKs7sssCb85Ermc5ONONPgkef3yHzKobfbg=";
  };

  format = "pyproject";

  nativeBuildInputs = [
    setuptools
    jaxlib
  ];

  propagatedBuildInputs = [
    jax
    typing-extensions
  ];

  nativeCheckInputs = [
    pytestCheckHook
    pytest-benchmark
    optax
  ];

  pythonImportsCheck = [
    "pytreeclass"
  ];

  meta = with lib; {
    description = "Visualize, create, and operate on pytrees";
    homepage = "https://pytreeclass.readthedocs.io/en/latest";
    license = licenses.asl20;
    maintainers = with maintainers; [ stephen-huan ];
  };
}
