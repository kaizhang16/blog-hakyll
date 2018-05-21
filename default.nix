let
  pkgs = import <nixpkgs> { };

in
  rec {
    blog = pkgs.haskellPackages.callPackage ./blog.nix { };
  }
