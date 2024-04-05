{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
  };

  outputs = {
    nixpkgs,
    flake-utils,
    rust-overlay,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      overlays = [(import rust-overlay)];
      pkgs = import nixpkgs {inherit system overlays;};

      runCiLocally = pkgs.writeScriptBin "ci-local" ''
        echo "Checking Rust formatting..."
        cargo fmt --check

        echo "Checking clippy..."
        cargo clippy --all-targets

        echo "Checking spelling..."
        codespell \
          --skip target,.git \
          --ignore-words-list crate

        echo "Testing Rust code..."
        cargo test
      '';

      nativeBuildInputs = with pkgs; [];
      buildInputs =
        [runCiLocally]
        ++ (with pkgs; [
          # Rust stuff, some stuff dev-only
          (rust-bin.nightly.latest.default.override {
            extensions = ["rust-src" "rust-analyzer"];
          })

          # Linting support
          alejandra
          codespell
        ]);
    in
      with pkgs; {
        devShells.default = mkShell {inherit buildInputs nativeBuildInputs;};
      });
}
