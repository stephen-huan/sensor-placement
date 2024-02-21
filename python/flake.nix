{
  description = "sensor";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;
      systems = lib.systems.flakeExposed;
      eachDefaultSystem = f: builtins.foldl' lib.attrsets.recursiveUpdate { }
        (map f systems);
    in
    eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python' = pkgs.python3.withPackages (ps: with ps; [
          cython_3
          jax
          jaxlib
          matplotlib
          scikit-learn
          setuptools
        ]);
        formatters = [ pkgs.black pkgs.isort ];
        linters = [ pkgs.nodePackages.pyright pkgs.ruff ];
      in
      {
        devShells.${system}.default = pkgs.mkShell {
          packages = [
            python'
            # pkgs.mkl
          ]
          ++ formatters
          ++ linters;
        };
      }
    );
}
