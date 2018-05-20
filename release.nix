let
  pkgs = import <nixpkgs> { };
in
  { blog = pkgs.haskellPackages.callPackage ./default.nix { };
  }
