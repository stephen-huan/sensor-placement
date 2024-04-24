{ lib
, fetchFromGitHub
, buildPythonPackage
, hatchling
, hatch-vcs
, beartype
, typing-extensions
, pytestCheckHook
, numpy
, ipython
}:

buildPythonPackage {
  pname = "cola_plum_dispatch";
  version = "0.1.4";

  # PyPi source doesn't contain tests
  src = fetchFromGitHub {
    owner = "mfinzi";
    repo = "plum";
    rev = "34bbcd14bb03dc3e006751015f289acc3b697335";
    sha256 = "sha256-ZyWWan6EpM3s8n+oIcBRfFmY831cS+FqF3/FJIJ1AG8=";
  };

  format = "pyproject";

  nativeBuildInputs = [
    hatchling
    hatch-vcs
  ];

  propagatedBuildInputs = [
    beartype
    typing-extensions
  ];

  nativeCheckInputs = [
    pytestCheckHook
    numpy
    ipython
  ];

  disabledTests = [
    "test_cache_class"
    "test_dispatch_multi"
    "test_abstract"
    "test_register"
    "test_call_mro"
    "test_call_object"
    "test_call_type"
    "test_register"
    "test_len"
  ];

  disabledTestPaths = [
    "tests/advanced"
  ];

  pythonImportsCheck = [
    "plum"
  ];

  meta = with lib; {
    description = "Multiple dispatch in Python";
    homepage = "https://github.com/mfinzi/plum";
    license = licenses.mit;
    maintainers = with maintainers; [ stephen-huan ];
  };
}
