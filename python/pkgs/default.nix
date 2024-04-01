{ pkgs }:

let
  inherit (pkgs.python3Packages) callPackage;
in
rec {
  cola-ml = callPackage ./cola-ml {
    inherit cola-plum-dispatch optree pytreeclass;
  };
  chex = callPackage ./chex { };
  cola-plum-dispatch = callPackage ./cola-plum-dispatch { };
  flax = callPackage ./flax { inherit optax orbax-checkpoint; };
  gpjax = callPackage ./gpjax {
    inherit cola-ml flax optax orbax-checkpoint;
    jaxopt = pkgs.python3Packages.jaxopt.override { inherit optax; };
    simple-pytree = simple-pytree_0_1_7;
  };
  optax = callPackage ./optax { inherit chex; };
  optree = callPackage ./optree { };
  # see tensorflow-build in pkgs/top-level/python-packages.nix
  orbax-checkpoint = (
    callPackage ./orbax-checkpoint { }
  ).override {
    protobuf = pkgs.python3Packages.protobuf.override {
      protobuf = pkgs.protobuf_21.override {
        abseil-cpp = pkgs.abseil-cpp_202301;
      };
    };
  };
  pytreeclass = callPackage ./pytreeclass { inherit optax; };
  simple-pytree = callPackage ./simple-pytree { inherit flax; };
  simple-pytree_0_1_7 = simple-pytree.overridePythonAttrs rec {
    version = "0.1.7";
    src = pkgs.fetchFromGitHub {
      owner = "cgarciae";
      repo = "simple-pytree";
      rev = version;
      sha256 = "sha256-Pss7LUnH8u/QQI+amnlKbqyc8tq8XNpcDJ6541pQxUw=";
    };
  };
}
