{ lib
, fetchFromGitHub
, buildPythonPackage
, cmake
, setuptools
, pybind11
, typing-extensions
, pytestCheckHook
, jax
, jaxlib
, numpy
, torch
}:

buildPythonPackage rec {
  pname = "optree";
  version = "0.10.0";

  # PyPi source doesn't contain tests/helpers.py
  src = fetchFromGitHub {
    owner = "metaopt";
    repo = "optree";
    rev = "v${version}";
    sha256 = "sha256-T0d9P0N3hN6NRCzu1AcjHWf5kdROC1CVzoV6dqakfOA=";
  };

  format = "pyproject";

  # hack: dontUseCmakeBuildDir = true doesn't work
  # https://discourse.nixos.org/t/27705
  dontUseCmakeConfigure = true;

  nativeBuildInputs = [
    setuptools
    pybind11
    cmake
  ];

  propagatedBuildInputs = [
    typing-extensions
  ];

  nativeCheckInputs = [
    pytestCheckHook
    # tests/integration
    jax
    jaxlib
    numpy
    torch
  ];

  # https://github.com/NixOS/nixpkgs/issues/255262
  preCheck = ''
    cd tests
  '';

  pythonImportsCheck = [
    "optree"
  ];

  meta = with lib; {
    description = "OpTree: Optimized PyTree Utilities";
    homepage = "https://optree.readthedocs.io/en/latest";
    license = licenses.asl20;
    maintainers = [ ];
  };
}
