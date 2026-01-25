{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  name = "bird-ai-env";

  buildInputs = with pkgs; [
    python310
    python310Packages.pip
    python310Packages.virtualenv
    
    # Audio Libraries required by librosa/soundfile
    libsndfile
    ffmpeg
    
    # CUDA dependencies for NixOS
    cudatoolkit
    linuxPackages.nvidia_x11
  ];

  shellHook = ''
    # Create venv if not exists
    if [ ! -d ".venv" ]; then
      echo "Creating virtual environment..."
      python -m venv .venv
    fi
    source .venv/bin/activate
    
    # Fix for library paths on NixOS so Python can find libsndfile and CUDA
    export LD_LIBRARY_PATH="${pkgs.libsndfile.out}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.cudatoolkit}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
    
    echo "🐦 Bird Environment Ready. GPU: GeForce RTX 2060"
  '';
}
