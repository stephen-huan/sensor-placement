{
  description = "sensor-placement";

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
        packages = import ./pkgs { inherit pkgs; };
        inherit (self.legacyPackages.${system}) python;
        python' = python.withPackages (ps: with ps; [
          cython_3
          gpjax
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
        legacyPackages.${system} = rec {
          # https://nixos.org/manual/nixpkgs/stable/#overriding-python-packages
          python = packages.python.override {
            packageOverrides = final: prev: packages;
            self = python;
          };
          python3Packages = packages // { inherit python; };
        };

        packages.${system} = {
          inherit (packages)
            cola-ml
            cola-plum-dispatch
            gpjax
            optree
            pytreeclass
            simple-pytree;
        };

        devShells.${system}.default = pkgs.mkShell {
          packages = [
            pkgs.nix-update
            python'
            # pkgs.mkl
          ]
          ++ formatters
          ++ linters;
        };
      }
    );
}
