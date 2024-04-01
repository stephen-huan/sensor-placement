# https://github.com/NixOS/nixpkgs/pull/297146
{ lib
, buildPythonPackage
, pythonOlder
, fetchFromGitHub
, absl-py
, cloudpickle
, dm-tree
, jax
, jaxlib
, numpy
, pytestCheckHook
, toolz
, typing-extensions
}:

buildPythonPackage rec {
  pname = "chex";
  version = "0.1.86";
  format = "setuptools";

  disabled = pythonOlder "3.9";

  src = fetchFromGitHub {
    owner = "deepmind";
    repo = "chex";
    rev = "refs/tags/v${version}";
    hash = "sha256-Z5Ns4fG5pC99I4xdGjDMKX6YZpTtd1y0TWcIOtr7dug=";
  };

  propagatedBuildInputs = [
    absl-py
    jaxlib
    jax
    numpy
    toolz
    typing-extensions
  ];

  pythonImportsCheck = [
    "chex"
  ];

  nativeCheckInputs = [
    cloudpickle
    dm-tree
    pytestCheckHook
  ];

  meta = with lib; {
    description = "Chex is a library of utilities for helping to write reliable JAX code.";
    homepage = "https://github.com/deepmind/chex";
    changelog = "https://github.com/google-deepmind/chex/releases/tag/v${version}";
    license = licenses.asl20;
    maintainers = with maintainers; [ ndl ];
  };
}
