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
          config.allowUnfree = true; # Wichtig für CUDA
        };
        
        python = pkgs.python310;
        
        # Diese C-Bibliotheken werden von pip-Paketen (PyTorch, OpenCV, Librosa) benötigt
        lib-path = with pkgs; lib.makeLibraryPath[
          stdenv.cc.cc.lib
          zlib
          glib
          libsndfile
          ffmpeg
          cudaPackages.cudatoolkit
          cudaPackages.cudnn
          # Wird für opencv-python benötigt
          xorg.libX11
          xorg.libXext
          xorg.libSM
          xorg.libICE
          xorg.libXrender
        ];

        # Komplettes LaTeX-Paket für main.tex
        tex = pkgs.texlive.combine {
          inherit (pkgs.texlive) scheme-full;
        };

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs =[
            # Python Tools
            python
            pkgs.python310Packages.pip
            pkgs.python310Packages.virtualenv
            
            # System Libs für Audio
            pkgs.libsndfile
            pkgs.ffmpeg
            
            # LaTeX Tools
            tex
            pkgs.latexmk
          ];

          # Dieser Hook wird ausgeführt, wenn du 'nix develop' eintippst
          shellHook = ''
            # Setze Bibliothekspfade für pip-Pakete unter NixOS
            export LD_LIBRARY_PATH="${lib-path}:$LD_LIBRARY_PATH"
            
            # Virtuelle Umgebung (.venv) erstellen, falls nicht vorhanden
            if[ ! -d ".venv" ]; then
              echo "🐣 Erstelle neue Python Virtual Environment (.venv)..."
              ${python}/bin/python -m venv .venv
            fi
            
            # Venv aktivieren
            source .venv/bin/activate
            
            echo ""
            echo "🐦 Piepmatz Polizei Nix-Environment geladen!"
            echo "================================================"
            echo "👉 Python: $(python --version)"
            echo "👉 Pip Install: pip install -r requirements.txt"
            echo "👉 Live-LaTeX: latexmk -pdf -pvc main.tex"
            echo "================================================"
          '';
        };
      }
    );
}
