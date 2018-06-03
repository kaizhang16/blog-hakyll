{ mkDerivation, base, hakyll, hakyll-sass, pandoc, pandoc-citeproc
, pandoc-crossref, pandoc-types, stdenv
}:
mkDerivation {
  pname = "blog";
  version = "1.2.0";
  src = ./.;
  isLibrary = false;
  isExecutable = true;
  executableHaskellDepends = [
    base hakyll hakyll-sass pandoc pandoc-citeproc pandoc-crossref
    pandoc-types
  ];
  license = stdenv.lib.licenses.bsd3;
}
