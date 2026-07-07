#!/bin/bash
# =========================================================================
# Installation and Launcher Generation Script for SSN Viewer & Tools (Linux/macOS)
# =========================================================================

# Resolve the real path of this script, following symbolic links
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

# Project root is two levels up from src/bin/
PROJECT_ROOT="$( cd -P "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd )"

cd "$PROJECT_ROOT"

echo "Setting up launchers for SSN Viewer & SSN Tools..."
echo "Project root: $PROJECT_ROOT"

# 1. Make sure all executables in src/bin have execution permissions
chmod +x src/bin/*.sh
echo "[OK] Configured execution permissions for scripts in src/bin/"

# 2. Create symbolic links in the project root pointing to the actual launchers
ln -sf src/bin/SSN_Viewer.sh SSN_Viewer
ln -sf src/bin/SSN_Tools.sh SSN_Tools
chmod +x SSN_Viewer SSN_Tools
echo "[OK] Created SSN_Viewer and SSN_Tools executables in project root."

# 3. Platform Specific Launcher Enhancements
OS_TYPE=$(uname)
if [ "$OS_TYPE" = "Linux" ]; then
    echo ""
    echo "Linux detected. Generating .desktop entry launchers..."

    # Define icon files
    VIEWER_ICON="${PROJECT_ROOT}/src/bin/logos/viewer_logo_large.png"
    TOOL_ICON="${PROJECT_ROOT}/src/bin/logos/tool_logo_large.png"

    # Create SSN_Viewer.desktop
    cat <<EOF > SSN_Viewer.desktop
[Desktop Entry]
Type=Application
Name=SSN Viewer
Comment=Sequence Similarity Network Viewer
Exec="${PROJECT_ROOT}/src/bin/SSN_Viewer.sh"
Path=${PROJECT_ROOT}
Icon=${VIEWER_ICON}
Terminal=true
Categories=Science;Biology;
EOF
    chmod +x SSN_Viewer.desktop

    # Create SSN_Tools.desktop
    cat <<EOF > SSN_Tools.desktop
[Desktop Entry]
Type=Application
Name=SSN Tools
Comment=Sequence Similarity Network Utilities
Exec="${PROJECT_ROOT}/src/bin/SSN_Tools.sh"
Path=${PROJECT_ROOT}
Icon=${TOOL_ICON}
Terminal=true
Categories=Science;Biology;
EOF
    chmod +x SSN_Tools.desktop

    echo "[OK] Created SSN_Viewer.desktop and SSN_Tools.desktop in project root using large logo icons."

    # Ask if they want to copy desktop files to Applications menu
    read -p "Would you like to install these launchers to your system application menu? (y/n): " install_menu
    if [[ "$install_menu" =~ ^[Yy]$ ]]; then
        mkdir -p ~/.local/share/applications
        cp SSN_Viewer.desktop ~/.local/share/applications/
        cp SSN_Tools.desktop ~/.local/share/applications/
        echo "[OK] Launchers successfully added to your system applications menu!"
    fi

    # Ask if they want to copy desktop files to Desktop
    if [ -d "$HOME/Desktop" ]; then
        read -p "Would you like to copy these launchers to your Desktop? (y/n): " install_desktop
        if [[ "$install_desktop" =~ ^[Yy]$ ]]; then
            cp SSN_Viewer.desktop "$HOME/Desktop/"
            cp SSN_Tools.desktop "$HOME/Desktop/"
            chmod +x "$HOME/Desktop/SSN_Viewer.desktop" "$HOME/Desktop/SSN_Tools.desktop"
            echo "[OK] Launchers successfully copied to your Desktop!"
        fi
    fi

elif [ "$OS_TYPE" = "Darwin" ]; then
    echo ""
    echo "macOS detected. Native symbolic link launchers created in project root."
    echo "To set a custom icon on macOS:"
    echo "  1. Right-click on the 'SSN_Viewer' or 'SSN_Tools' launcher in Finder and select 'Get Info'."
    echo "  2. Open the corresponding large logo in Preview (e.g. 'src/bin/logos/viewer_logo_large.png' or 'src/bin/logos/tool_logo_large.png'), press Cmd+A, then Cmd+C to copy it."
    echo "  3. Click on the file icon thumbnail at the top-left of the 'Get Info' window and press Cmd+V to paste."
fi

echo ""
echo "Setup Complete! You can now run SSN Viewer and Tools using the launchers in the project root."
