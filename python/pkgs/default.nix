{ pkgs }:

pkgs.python3Packages.overrideScope (final: prev: {
  chex = final.callPackage ./chex { };
  cola-ml = final.callPackage ./cola-ml { };
  cola-plum-dispatch = final.callPackage ./cola-plum-dispatch { };
  # https://github.com/NixOS/nixpkgs/pull/297146
  flax = prev.flax.overridePythonAttrs (previousAttrs: {
    disabledTestPaths = previousAttrs.disabledTestPaths or [ ] ++ [
      "flax/experimental/nnx/examples/*"
    ];
  });
  gpjax = final.callPackage ./gpjax {
    simple-pytree = final.simple-pytree.overridePythonAttrs rec {
      version = "0.1.7";
      src = pkgs.fetchFromGitHub {
        owner = "cgarciae";
        repo = "simple-pytree";
        rev = version;
        sha256 = "sha256-Pss7LUnH8u/QQI+amnlKbqyc8tq8XNpcDJ6541pQxUw=";
      };
    };
  };
  optax = final.callPackage ./optax { };
  optree = final.callPackage ./optree { };
  # see tensorflow-build in pkgs/top-level/python-packages.nix
  orbax-checkpoint = prev.orbax-checkpoint.override {
    protobuf = pkgs.python3Packages.protobuf.override {
      protobuf = pkgs.protobuf_21.override {
        abseil-cpp = pkgs.abseil-cpp_202301;
      };
    };
  };
  pytreeclass = final.callPackage ./pytreeclass { };
  simple-pytree = final.callPackage ./simple-pytree { };
})
