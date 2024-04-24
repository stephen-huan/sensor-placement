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
          cython
          gpjax
          jax
          jaxlib
          matplotlib
          scikit-learn
          setuptools
        ]);
        formatters = [ pkgs.black pkgs.isort pkgs.nixpkgs-fmt ];
        linters = [ pkgs.nodePackages.pyright pkgs.ruff pkgs.statix ];
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

        formatter.${system} = pkgs.writeShellApplication {
          name = "formatter";
          runtimeInputs = formatters;
          text = ''
            isort "$@"
            black "$@"
            nixpkgs-fmt "$@"
          '';
        };

        checks.${system}.lint = pkgs.stdenvNoCC.mkDerivation {
          name = "lint";
          src = ./.;
          doCheck = true;
          nativeCheckInputs = formatters ++ linters ++ lib.singleton python';
          checkPhase = ''
            isort --check --diff .
            black --check --diff .
            nixpkgs-fmt --check .
            ruff check .
            pyright .
            statix check
          '';
          installPhase = "touch $out";
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
