{
  description = "Piepmatz Polizei - Machine Learning & LaTeX Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        
        # Python 3.12 ist aktuell der "Sweetspot" für Machine Learning auf NixOS
        python = pkgs.python312;
        
        lib-path = with pkgs; lib.makeLibraryPath[
          stdenv.cc.cc.lib
          zlib
          glib
          libsndfile
          ffmpeg
          cudaPackages.cudatoolkit
          cudaPackages.cudnn
          libX11
          libXext
          libSM
          libICE
          libXrender
	  portaudio
        ];

        tex = pkgs.texlive.combine {
          inherit (pkgs.texlive) scheme-full;
        };

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs =[
            python
            python.pkgs.pip         # Nutzt exakt die Pip-Version von Python 3.12
            python.pkgs.virtualenv
            pkgs.libsndfile
            pkgs.ffmpeg
            tex
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${lib-path}:$LD_LIBRARY_PATH"
            
            if [ ! -d ".venv" ]; then
              echo "🐣 Erstelle neue Python Virtual Environment (.venv)..."
              ${python}/bin/python -m venv .venv
            fi
            
            source .venv/bin/activate
            
            echo ""
            echo "🐦 Piepmatz Polizei Nix-Environment geladen (Python 3.12)!"
          '';
        };
      }
    );
}
