#!/bin/bash
# Install Google Cloud CLI for Phase C deployment

echo "════════════════════════════════════════════════════════"
echo "  Google Cloud CLI Installation"
echo "════════════════════════════════════════════════════════"
echo ""

# Check if already installed
if command -v gcloud &> /dev/null; then
    echo "✅ gcloud CLI is already installed"
    gcloud version
    exit 0
fi

echo "Installing Google Cloud CLI via snap (requires sudo)..."
echo ""

# Install via snap (recommended)
sudo snap install google-cloud-cli --classic

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Installation successful!"
    echo ""
    echo "Next steps:"
    echo "1. Authenticate: gcloud auth login"
    echo "2. Set project: gcloud config set project YOUR_PROJECT_ID"
    echo "3. Launch Phase C: cd phase_c/gcp && ./launch_phaseC.sh"
else
    echo ""
    echo "❌ Installation failed. Try alternative method:"
    echo ""
    echo "curl https://sdk.cloud.google.com | bash"
    echo "exec -l \$SHELL"
    echo "gcloud init"
fi
