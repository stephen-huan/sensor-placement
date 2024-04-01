# https://github.com/NixOS/nixpkgs/pull/266467
{ lib
, buildPythonPackage
, pythonOlder
, fetchPypi
, flit-core
, absl-py
, etils
, importlib-resources
, jax
, jaxlib
, msgpack
, nest-asyncio
, numpy
, protobuf
, pyyaml
, tensorstore
, typing-extensions
, flax
, pytest
, pytest-xdist
, pytestCheckHook
}:

let orbax-checkpoint = buildPythonPackage rec {
  pname = "orbax-checkpoint";
  version = "0.5.7";
  pyproject = true;

  disabled = pythonOlder "3.9";

  src = fetchPypi {
    pname = "orbax_checkpoint";
    inherit version;
    hash = "sha256-3hRUm4mSIKT0RUU5Z8GsLXFluBUlM0JYd0YAXwOpgTs=";
  };

  nativeBuildInputs = [
    flit-core
  ];

  propagatedBuildInputs = [
    absl-py
    etils
    jax
    jaxlib
    importlib-resources
    msgpack
    nest-asyncio
    numpy
    protobuf
    pyyaml
    tensorstore
    typing-extensions
  ];

  nativeCheckInputs = [
    flax
    pytest-xdist
    pytestCheckHook
  ];

  preCheck = ''
    export HOME=$(mktemp -d)
  '';

  doCheck = false;

  # Tests depend on flax, which depends on this package.
  # To avoid infinite recursion, we only enable tests when building passthru.tests.
  passthru.tests = {
    check = orbax-checkpoint.overridePythonAttrs (_: { doCheck = true; });
  };

  pythonImportsCheck = [ "orbax.checkpoint" ];

  meta = with lib; {
    description = "A checkpointing library oriented towards JAX users";
    homepage = "https://pypi.org/project/orbax-checkpoint";
    changelog = "https://github.com/google/orbax/blob/main/checkpoint/CHANGELOG.md";
    license = licenses.asl20;
    maintainers = with maintainers; [ GaetanLepage ];
  };
};
in orbax-checkpoint
