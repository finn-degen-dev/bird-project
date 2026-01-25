{
  description = "Piepmatz Polizei - Jugend forscht Dokumentation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Ein komplettes LaTeX-Paket, damit keine dependencies fehlen
        tex = pkgs.texlive.combine {
            inherit (pkgs.texlive) scheme-full;
        };
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "jugend-forscht-arbeit";
          src = ./.; # Kopiert das aktuelle Verzeichnis in den Build-Context

          buildInputs = [ 
            tex 
            pkgs.coreutils 
          ];

          # Wir gehen davon aus, dass deine Hauptdatei 'main.tex' heißt
          buildPhase = ''
            # Sicherstellen, dass visuals/ existiert (wird durch src=./. kopiert)
            echo "Kompiliere Dokumentation..."
            
            # latexmk sorgt für autom. BibTex, Indizes und mehrfaches Kompilieren
            latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
          '';

          installPhase = ''
            mkdir -p $out
            cp main.pdf $out/Piepmatz_Polizei_Arbeit.pdf
          '';
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [ 
            tex
            pkgs.latexmk
          ];
          
          shellHook = ''
            echo "Piepmatz Polizei LaTeX Shell"
            echo "Nutze 'latexmk -pdf -pvc main.tex' für Live-Preview."
          '';
        };
      }
    );
}
