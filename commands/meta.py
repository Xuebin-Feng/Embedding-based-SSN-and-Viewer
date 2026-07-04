import Command_Engine
import os
import re
import numpy as np
import pandas as pd
import SSN_Utils as utils
from PyQt6 import QtWidgets, QtCore
from PyQt6 import QtGui


class MetadataTableModel(QtCore.QAbstractTableModel):
    """High-performance table model backed directly by viewer.metadata arrays."""
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.columns = []
        self.refresh_columns()

    def refresh_columns(self):
        self.beginResetModel()
        self.columns = ["Node ID"]
        if hasattr(self.viewer, 'metadata'):
            self.columns.extend([k for k in self.viewer.metadata.keys() if k.lower() != "length"])
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return getattr(self.viewer, 'n_nodes', 0)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.columns)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()
        col_name = self.columns[col]
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if col_name == "Node ID":
                return str(self.viewer.full_headers[row])
            else:
                meta_entry = self.viewer.metadata.get(col_name)
                if meta_entry is not None:
                    val = meta_entry["values"][row]
                    if isinstance(val, (float, np.floating)):
                        if pd.isna(val):
                            return ""
                        return f"{val:g}"
                    return str(val) if pd.notna(val) else ""
        elif role == QtCore.Qt.ItemDataRole.EditRole:
            if col_name == "Node ID":
                return str(self.viewer.full_headers[row])
            else:
                meta_entry = self.viewer.metadata.get(col_name)
                if meta_entry is not None:
                    val = meta_entry["values"][row]
                    if isinstance(val, (float, np.floating)):
                        if pd.isna(val):
                            return ""
                        return str(val)
                    return str(val) if pd.notna(val) else ""
        elif role == QtCore.Qt.ItemDataRole.UserRole:
            # Return raw sortable value
            if col_name == "Node ID":
                return str(self.viewer.full_headers[row])
            else:
                meta_entry = self.viewer.metadata.get(col_name)
                if meta_entry is not None:
                    val = meta_entry["values"][row]
                    if isinstance(val, (float, np.floating)) and pd.isna(val):
                        return None
                    return val
        return None

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags
        base_flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        col_name = self.columns[index.column()]
        if col_name != "Node ID":
            return base_flags | QtCore.Qt.ItemFlag.ItemIsEditable
        return base_flags

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != QtCore.Qt.ItemDataRole.EditRole:
            return False
        row = index.row()
        col = index.column()
        col_name = self.columns[col]
        
        if col_name == "Node ID":
            return False
            
        meta_entry = self.viewer.metadata.get(col_name)
        if meta_entry is None:
            return False
            
        prop_type = meta_entry["type"]
        if prop_type == "number":
            try:
                if not str(value).strip():
                    parsed_val = np.nan
                else:
                    parsed_val = float(value)
            except ValueError:
                return False
        else:
            parsed_val = str(value)
            
        meta_entry["values"][row] = parsed_val
        self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole])
        return True

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return self.columns[section]
            else:
                return str(section + 1)
        return None


class MultiColumnFilterProxyModel(QtCore.QSortFilterProxyModel):
    """Proxy model that filters by visibility mask AND per-column text filters."""
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.column_filters = {}  # col_index -> filter_text

    def set_column_filter(self, col, text):
        self.column_filters[col] = text.lower().strip()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        # Visibility mask filter
        if hasattr(self.viewer, 'visible_mask'):
            if not bool(self.viewer.visible_mask[source_row]):
                return False
        # Per-column text filters
        for col, text in self.column_filters.items():
            if not text:
                continue
            idx = self.sourceModel().index(source_row, col)
            val = str(self.sourceModel().data(idx, QtCore.Qt.ItemDataRole.DisplayRole) or "")
            if text not in val.lower():
                return False
        return True

    def lessThan(self, left, right):
        left_val = self.sourceModel().data(left, QtCore.Qt.ItemDataRole.UserRole)
        right_val = self.sourceModel().data(right, QtCore.Qt.ItemDataRole.UserRole)
        # Handle None
        if left_val is None and right_val is None:
            return False
        if left_val is None:
            return True
        if right_val is None:
            return False
        # Numeric comparison
        if isinstance(left_val, (int, float, np.integer, np.floating)) and isinstance(right_val, (int, float, np.integer, np.floating)):
            return float(left_val) < float(right_val)
        return str(left_val).lower() < str(right_val).lower()

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Vertical:
                return str(section + 1)
        return super().headerData(section, orientation, role)


class FilterHeaderView(QtWidgets.QHeaderView):
    """Custom header with filter QLineEdit widgets embedded below each column header."""
    filterChanged = QtCore.pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self._editors = []
        self._padding = 4
        self.setSectionsClickable(True)
        self.setSortIndicatorShown(True)
        self.sectionResized.connect(self._adjust_positions)
        self.sectionMoved.connect(self._adjust_positions)

    def setFilterBoxes(self, count):
        # Remove old editors
        for ed in self._editors:
            ed.deleteLater()
        self._editors = []
        for i in range(count):
            editor = QtWidgets.QLineEdit(self)
            editor.setPlaceholderText("Filter...")
            editor.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #d0d7de;
                    border-radius: 3px;
                    padding: 1px 4px;
                    font-size: 8.5pt;
                    background-color: #ffffff;
                }
                QLineEdit:focus {
                    border-color: #0969da;
                }
            """)
            editor.textChanged.connect(lambda text, col=i: self.filterChanged.emit(col, text))
            self._editors.append(editor)
        self._adjust_positions()

    def _adjust_positions(self):
        for i, editor in enumerate(self._editors):
            h = self.sectionSize(i)
            px = self.sectionPosition(i) - self.offset()
            filter_h = 22
            editor.setGeometry(px + self._padding, 2,
                               h - 2 * self._padding, filter_h)

    def paintSection(self, painter, rect, logicalIndex):
        painter.save()
        # Draw background
        painter.fillRect(rect, QtGui.QColor("#f0f0f0"))
        
        # Draw right and bottom borders
        painter.setPen(QtGui.QColor("#e2e2e2"))
        painter.drawLine(rect.right(), rect.top(), rect.right(), rect.bottom())
        painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())
        
        # Draw column title text in the lower half
        filter_h = 22
        offset_y = filter_h + 4
        text_rect = QtCore.QRect(rect.x() + 6, rect.y() + offset_y, rect.width() - 12, rect.height() - offset_y)
        
        # Get header text
        text = str(self.model().headerData(logicalIndex, QtCore.Qt.Orientation.Horizontal, QtCore.Qt.ItemDataRole.DisplayRole) or "")
        
        # Append sort arrow if this is the sorted column
        if self.isSortIndicatorShown() and self.sortIndicatorSection() == logicalIndex:
            order = self.sortIndicatorOrder()
            text += "  \u25B2" if order == QtCore.Qt.SortOrder.AscendingOrder else "  \u25BC"
        
        painter.setPen(QtGui.QColor("#1f2328"))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, text)
        
        painter.restore()

    def sizeHint(self):
        s = super().sizeHint()
        s.setHeight(s.height() + 26)  # Extra space for filter row
        return s

    def updateGeometries(self):
        super().updateGeometries()
        self._adjust_positions()

    def showEvent(self, event):
        super().showEvent(event)
        self._adjust_positions()


def print_help(meta_dir_help):
    print(f"""
    Node Metadata Manager Tool
    ==========================
    Usage:
      meta
          Opens a file explorer to manually select a metadata file (.xlsx, .xls, .csv)
          to upload and merge into the current viewer session.
      meta display/show <property_name>
          Displays the selected property of a node in the top-right corner of the window
          whenever a node is left-clicked.
      meta display/show clear/off
          Clears and removes the metadata property display.
      meta retrieve/download/export [filename] [expression]
          Downloads the current session metadata. If no filename is provided, 
          opens a save file dialog for location and name selection.
          An optional boolean logic expression can be provided at the end to filter 
          which nodes are exported (e.g., #cluster_1#, {{Length>500}}, or $sele$).
      meta help
          Displays this help message.

    Examples:
      meta
      meta show Organism
      meta show clear
      meta download
      meta download {{Length>500}}
      meta export filtered_meta.xlsx #cluster_1#
      meta export selected_meta.xlsx $sele$
    """)

def is_logic_expression(arg):
    # Check if the argument has typical boolean logic expression characters:
    # { } (metadata), # # (cluster/labels), @ @ (file search), " " (header text query),
    # &, |, !, ^ (logic operators), or represents UI selection $sele$
    if any(c in arg for c in '{}#@&|!^"'):
        return True
    if arg.lower() == '$sele$':
        return True
    # check AA position format (e.g. P106, _100)
    if re.match(r'^[a-zA-Z_][\d\.]+$', arg):
        return True
    return False

def inject_spreadsheet_panel(viewer, show_sidebar=True):
    if not getattr(viewer, 'metadata', None):
        return
    real_keys = [k for k in viewer.metadata.keys() if k.lower() != "length"]
    if not real_keys:
        return

    if hasattr(viewer, 'tab_widget'):
        tab_idx = -1
        for idx in range(viewer.tab_widget.count()):
            if viewer.tab_widget.tabText(idx) == "Metadata":
                tab_idx = idx
                break

        if tab_idx == -1:
            # Build the table view
            table_view = QtWidgets.QTableView()
            source_model = MetadataTableModel(viewer)
            proxy_model = MultiColumnFilterProxyModel(viewer)
            proxy_model.setSourceModel(source_model)
            table_view.setModel(proxy_model)

            table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            table_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
            table_view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
            table_view.setSortingEnabled(False)
            table_view.setAlternatingRowColors(True)

            table_view.setStyleSheet("""
                QTableView {
                    gridline-color: #e2e2e2;
                    background-color: #ffffff;
                    alternate-background-color: #f8f9fa;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 9.5pt;
                    border: none;
                }
                QTableView::item:selected {
                    background-color: #e6f7ff;
                    color: #1f2328;
                }
                QHeaderView::section {
                    background-color: #f0f0f0;
                    padding: 4px;
                    border: none;
                    border-right: 1px solid #e2e2e2;
                    border-bottom: 2px solid #e2e2e2;
                    font-weight: bold;
                    font-size: 9pt;
                }
            """)

            # Install filter header
            filter_header = FilterHeaderView(table_view)
            table_view.setHorizontalHeader(filter_header)
            filter_header.setFilterBoxes(source_model.columnCount())
            filter_header.filterChanged.connect(proxy_model.set_column_filter)

            # Custom Sorting Cycle: Ascending -> Descending -> Unsorted
            table_view._current_sort_col = -1
            table_view._current_sort_order = QtCore.Qt.SortOrder.AscendingOrder

            def handle_header_click(logical_index):
                header = table_view.horizontalHeader()
                if table_view._current_sort_col == logical_index:
                    if table_view._current_sort_order == QtCore.Qt.SortOrder.AscendingOrder:
                        # Cycle from Ascending to Descending
                        table_view._current_sort_order = QtCore.Qt.SortOrder.DescendingOrder
                        proxy_model.sort(logical_index, QtCore.Qt.SortOrder.DescendingOrder)
                        header.setSortIndicator(logical_index, QtCore.Qt.SortOrder.DescendingOrder)
                        header.setSortIndicatorShown(True)
                    else:
                        # Cycle from Descending to Unsorted
                        table_view._current_sort_col = -1
                        proxy_model.sort(-1, QtCore.Qt.SortOrder.AscendingOrder)
                        header.setSortIndicatorShown(False)
                else:
                    # Sort Ascending on new column
                    table_view._current_sort_col = logical_index
                    table_view._current_sort_order = QtCore.Qt.SortOrder.AscendingOrder
                    proxy_model.sort(logical_index, QtCore.Qt.SortOrder.AscendingOrder)
                    header.setSortIndicator(logical_index, QtCore.Qt.SortOrder.AscendingOrder)
                    header.setSortIndicatorShown(True)

            filter_header.sectionClicked.connect(handle_header_click)

            # Double-click to locate/pan (only for Node ID column)
            def on_table_double_clicked(index):
                col_name = source_model.columns[index.column()]
                if col_name == "Node ID":
                    source_index = proxy_model.mapToSource(index)
                    row = source_index.row()
                    if hasattr(viewer, 'pos') and row < len(viewer.pos):
                        viewer.view.camera.center = tuple(viewer.pos[row][:2])
                        viewer.selected_indices = [row]
                        viewer.selected_node_idx = row
                        viewer.update_selection_visual()
                        if hasattr(viewer, '_hud_timer'):
                            viewer._hud_timer.start()
            table_view.doubleClicked.connect(on_table_double_clicked)

            # Selection sync: table -> viewer
            def on_table_selection_changed():
                if getattr(viewer, '_syncing_selection', False):
                    return
                viewer._syncing_selection = True
                try:
                    selected_rows = table_view.selectionModel().selectedRows()
                    new_selected = []
                    for index in selected_rows:
                        source_index = proxy_model.mapToSource(index)
                        new_selected.append(source_index.row())
                    viewer.selected_indices = new_selected
                    viewer.update_selection_visual()
                finally:
                    viewer._syncing_selection = False
            table_view.selectionModel().selectionChanged.connect(on_table_selection_changed)

            # Store references
            viewer.metadata_table_view = table_view
            viewer.metadata_source_model = source_model
            viewer.metadata_proxy_model = proxy_model

            # Selection sync: viewer -> table
            def sync_selection_to_table(node_idx=None):
                if getattr(viewer, '_syncing_selection', False):
                    return
                viewer._syncing_selection = True
                try:
                    selection_model = table_view.selectionModel()
                    selection_model.clearSelection()
                    if node_idx is not None:
                        source_idx = source_model.index(node_idx, 0)
                        proxy_idx = proxy_model.mapFromSource(source_idx)
                        if proxy_idx.isValid():
                            row_start = proxy_model.index(proxy_idx.row(), 0)
                            row_end = proxy_model.index(proxy_idx.row(), source_model.columnCount() - 1)
                            selection_model.select(
                                QtCore.QItemSelection(row_start, row_end),
                                QtCore.QItemSelectionModel.SelectionFlag.Select
                            )
                            table_view.scrollTo(proxy_idx, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
                finally:
                    viewer._syncing_selection = False
            viewer.sync_metadata_table_selection = sync_selection_to_table

            # Visibility sync
            def sync_visibility_to_table():
                proxy_model.invalidateFilter()
            viewer.sync_metadata_table_visibility = sync_visibility_to_table

            # Add tab
            viewer.tab_widget.addTab(table_view, "Metadata")
            tab_idx = viewer.tab_widget.count() - 1

            # Sync initial selection
            sel_idx = getattr(viewer, 'selected_node_idx', None)
            if sel_idx is not None:
                sync_selection_to_table(sel_idx)

        else:
            # Tab already exists — refresh columns if needed
            if hasattr(viewer, 'metadata_source_model'):
                viewer.metadata_source_model.refresh_columns()
                if hasattr(viewer, 'metadata_table_view'):
                    hdr = viewer.metadata_table_view.horizontalHeader()
                    if isinstance(hdr, FilterHeaderView):
                        hdr.setFilterBoxes(viewer.metadata_source_model.columnCount())

        viewer.tab_widget.setCurrentIndex(tab_idx)
        if show_sidebar:
            viewer.set_sidebar_visible(True)
        else:
            viewer.set_sidebar_visible(False)

def run(viewer, args):
    # --- 1. Help & Usage ---
    import SSN_Config as cfg
    meta_dir = getattr(cfg, 'METADATA_DIR', os.path.join("Cache_Files", "Meta_Data"))
    os.makedirs(meta_dir, exist_ok=True)

    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help(meta_dir)
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    # --- 1.5. Display/Show Property ---
    if args and args[0].lower() in ['display', 'show']:
        if len(args) < 2:
            msg = "Usage: meta display/show <property_name> OR meta display/show clear/off"
            Command_Engine.print_help(viewer, msg)
            return

        prop_name = " ".join(args[1:]).strip()
        if prop_name.lower() in ['clear', 'off']:
            if 'meta_display' in viewer.hud_displays:
                viewer.hud_displays['meta_display'].hide()
            viewer.meta_display_prop = None
            msg = "Metadata display cleared."
            Command_Engine.print_help(viewer, msg)
            return

        if not prop_name:
            msg = "Error: Please specify a valid property name."
            Command_Engine.print_help(viewer, msg)
            return

        # Case-insensitive check in metadata
        available_props = list(viewer.metadata.keys()) if getattr(viewer, 'metadata', None) else []
        resolved_prop = None
        for p in available_props:
            if p.lower() == prop_name.lower():
                resolved_prop = p
                break

        if not resolved_prop:
            resolved_prop = prop_name
            if available_props:
                print(f"Warning: Property '{prop_name}' not found in current metadata. Available properties: {', '.join(available_props)}")

        viewer.meta_display_prop = resolved_prop

        # Register HUD display if not already present
        if 'meta_display' not in viewer.hud_displays:
            from SSN_Viewer import HUDDisplay
            
            class MetaHUDDisplay(HUDDisplay):
                def __init__(self, main_viewer):
                    super().__init__(
                        viewer=main_viewer,
                        name='meta_display',
                        pos_fn=lambda size: (size[0] - 30, 60),
                        anchor_x='right',
                        anchor_y='bottom'
                    )
                
                def on_node_clicked(self, node_idx):
                    p_name = getattr(self.viewer, 'meta_display_prop', None)
                    if p_name and getattr(self.viewer, 'metadata', None) and p_name in self.viewer.metadata:
                        val = self.viewer.metadata[p_name]["values"][node_idx]
                        import pandas as pd
                        val_str = str(val).strip() if pd.notna(val) and val is not None else "N/A"
                        self.show(f"{p_name}: {val_str}")
                    else:
                        self.show(f"{p_name}: N/A")

            viewer.hud_displays['meta_display'] = MetaHUDDisplay(viewer)

        # Update if a node is currently selected
        display = viewer.hud_displays['meta_display']
        node_idx = getattr(viewer, 'selected_node_idx', None)
        if node_idx is not None and getattr(viewer, 'metadata', None) and resolved_prop in viewer.metadata:
            val = viewer.metadata[resolved_prop]["values"][node_idx]
            val_str = str(val).strip() if pd.notna(val) and val is not None else "N/A"
            display.show(f"{resolved_prop}: {val_str}")
        else:
            display.show(f"{resolved_prop}: -")

        msg = f"Metadata display enabled for property: '{resolved_prop}'"
        Command_Engine.print_help(viewer, msg)
        return

    # --- 2. Resolve Directory & Path ---
    is_download = False
    filepath = None
    filename = ""
    expr = None

    # Check if download mode is requested
    if args and args[0].lower() in ['retrieve', 'download', 'export']:
        is_download = True
        
        # Check if filename or expression is provided
        if len(args) >= 2:
            if len(args) >= 3 and is_logic_expression(args[1]) and not is_logic_expression(args[-1]):
                # Form: download <expression> <filename>
                filename = args[-1]
                expr = " ".join(args[1:-1]).strip()
            elif is_logic_expression(args[1]):
                # Form: download <expression> (filename will be selected via dialog)
                expr = " ".join(args[1:]).strip()
            else:
                # Form: download <filename> [expression]
                filename = args[1]
                if len(args) >= 3:
                    expr = " ".join(args[2:]).strip()

        if filename:
            # Enforce default .xlsx extension if no extension is specified
            _, ext = os.path.splitext(filename)
            if not ext:
                filename += ".xlsx"
                ext = ".xlsx"
            filepath = os.path.join(meta_dir, filename)
        else:
            # Open save file dialog if no filename provided for download
            try:
                file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    viewer.canvas.native,
                    "Save Metadata File",
                    meta_dir,
                    "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
                )
                if not file_path:
                    msg = "Metadata export cancelled."
                    Command_Engine.print_help(viewer, msg)
                    return
                filepath = file_path
                filename = os.path.basename(file_path)
                _, ext = os.path.splitext(filename)
            except Exception as e:
                msg = f"Error opening save dialog: {e}"
                Command_Engine.print_help(viewer, msg)
                return
    else:
        # Upload mode
        if args:
            file_paths = []
            for arg in args:
                path = arg.strip()
                # 1. Check if direct path exists
                if os.path.exists(path):
                    file_paths.append(os.path.abspath(path))
                else:
                    # 2. Check if it exists inside the metadata directory
                    path_in_dir = os.path.join(meta_dir, path)
                    if os.path.exists(path_in_dir):
                        file_paths.append(os.path.abspath(path_in_dir))
                    else:
                        # 3. Try appending common extensions
                        found = False
                        for ext in ['.xlsx', '.xls', '.csv']:
                            if os.path.exists(path + ext):
                                file_paths.append(os.path.abspath(path + ext))
                                found = True
                                break
                            elif os.path.exists(os.path.join(meta_dir, path + ext)):
                                file_paths.append(os.path.abspath(os.path.join(meta_dir, path + ext)))
                                found = True
                                break
                        if not found:
                            msg = f"Error: Metadata file '{path}' not found (checked absolute, relative, and {meta_dir})."
                            Command_Engine.print_help(viewer, msg)
                            return
        else:
            # No arguments -> Open file selection dialog (multiple files allowed)
            try:
                file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                    viewer.canvas.native,
                    "Select Metadata Files",
                    meta_dir,
                    "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*)"
                )
                if not file_paths:
                    msg = "Metadata upload cancelled."
                    Command_Engine.print_help(viewer, msg)
                    return
            except Exception as e:
                msg = f"Error opening file dialog: {e}"
                Command_Engine.print_help(viewer, msg)
                return

    # --- 3. Execute Download Mode ---
    if is_download:
        if not getattr(viewer, 'metadata', None):
            msg = "Error: No metadata available in the viewer to download."
            Command_Engine.print_help(viewer, msg)
            return

        try:
            # Evaluate logic expression if provided
            mask = np.ones(viewer.n_nodes, dtype=bool)
            if expr:
                # Update the selection file in case the expression references $sele$
                header_dir = getattr(cfg, 'HEADER_LIST_DIR', os.path.join("Input_Files", "Header_Lists"))
                os.makedirs(header_dir, exist_ok=True)
                sele_path = os.path.join(header_dir, "_sele.txt")
                if hasattr(viewer, 'selected_indices') and viewer.selected_indices:
                    with open(sele_path, "w", encoding="utf-8") as f:
                        for idx in viewer.selected_indices:
                            f.write(viewer.full_headers[idx] + "\n")
                else:
                    if os.path.exists(sele_path):
                        open(sele_path, 'w').close()

                # Preprocess expression
                expr_cleaned = re.sub(r'["\']?\$sele\$["\']?', '@_sele.txt@', expr, flags=re.IGNORECASE)
                expr_cleaned = re.sub(r'\{([^}]+)\}', lambda m: '{' + m.group(1).replace(' ', '') + '}', expr_cleaned)
                
                viewer_to_aln = np.full(len(viewer.full_headers), -1, dtype=int)
                if (getattr(viewer, 'alignment', None).aln if getattr(viewer, 'alignment', None) else None) is not None:
                    for i, h in enumerate(viewer.full_headers):
                        if h in viewer.alignment.seq_map:
                            viewer_to_aln[i] = viewer.alignment.seq_map[h]
                valid_indices = np.where(viewer_to_aln != -1)[0]
                
                mask = Command_Engine.parse_advanced_expression(
                    expr_cleaned,
                    viewer_to_aln,
                    valid_indices,
                    viewer.full_headers,
                    getattr(viewer, 'cluster_labels', None),
                    getattr(viewer, 'group_labels', None),
                    getattr(viewer, 'alignment', None),
                    metadata=viewer.metadata
                )
                
                if np.sum(mask) == 0:
                    msg = f"Error: No nodes matched the expression '{expr}'."
                    Command_Engine.print_help(viewer, msg)
                    return

            # Build spreadsheet layout
            prop_names = [p for p in viewer.metadata.keys() if p != "Length"]
            
            row_0 = [""] + prop_names
            row_1 = [""] + [viewer.metadata[p]["type"] for p in prop_names]
            
            rows = [row_0, row_1]
            for i in range(viewer.n_nodes):
                if not mask[i]:
                    continue
                has_valid_prop = False
                row_val = [viewer.full_headers[i]]
                for p in prop_names:
                    val = viewer.metadata[p]["values"][i]
                    if viewer.metadata[p]["type"] == "number":
                        if pd.notna(val):
                            has_valid_prop = True
                            row_val.append(val)
                        else:
                            row_val.append("")
                    else:
                        if val is not None and str(val).strip() != "":
                            has_valid_prop = True
                            row_val.append(val)
                        else:
                            row_val.append("")
                
                # Only include nodes that have at least one populated property,
                # unless they were explicitly selected via expression filter.
                if has_valid_prop or expr:
                    rows.append(row_val)

            df = pd.DataFrame(rows)

            if ext.lower() == ".csv":
                df.to_csv(filepath, header=False, index=False)
            else:
                df.to_excel(filepath, header=False, index=False)

            msg = f"Metadata successfully downloaded to {filepath}"
            if expr:
                msg += f" (filtered by: {expr})"
            Command_Engine.print_help(viewer, msg)
        except Exception as e:
            msg = f"Error downloading metadata: {e}"
            Command_Engine.print_help(viewer, msg)
        return

    # --- 4. Execute Upload Mode ---
    successful_files = []
    failed_files = []
    matched_nodes = set()
    total_unmatched = 0
    all_merged_props = set()

    # Safe save state for undo support (save once before all uploads)
    viewer._save_state()

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filename)
        if not os.path.exists(filepath):
            failed_files.append((filename, "File not found."))
            continue

        try:
            # Load the file
            if ext.lower() == ".csv":
                df = pd.read_csv(filepath, header=None, dtype=str)
            else:
                df = pd.read_excel(filepath, header=None)

            # Check dimensions
            if df.shape[0] < 3 or df.shape[1] < 2:
                raise ValueError("Invalid file format. Must contain at least sequence headers and one property column.")

            # Extract property names and types from first 2 rows (columns 1 onwards)
            prop_names = []
            valid_cols = []
            for col_idx in range(1, df.shape[1]):
                val = df.iloc[0, col_idx]
                if pd.notna(val) and str(val).strip():
                    prop_names.append(str(val).strip())
                    valid_cols.append(col_idx)

            if not prop_names:
                raise ValueError("No valid property names found in the first row.")

            # Validate property names to ensure they only contain alphanumeric characters, underscores, hyphens, and periods
            illegal_props = [prop for prop in prop_names if not re.match(r'^[a-zA-Z0-9_\-\.]+$', prop)]
            if illegal_props:
                raise ValueError(
                    f"Property names {', '.join([repr(p) for p in illegal_props])} contain illegal characters. "
                    "Allowed characters are: letters, numbers, underscores (_), hyphens (-), and periods (.)"
                )

            prop_types = []
            for col_idx in valid_cols:
                val = df.iloc[1, col_idx]
                if pd.notna(val) and str(val).strip():
                    t = str(val).strip().lower()
                    if t in ['number', 'num', 'numerical']:
                        prop_types.append('number')
                    else:
                        prop_types.append('text')
                else:
                    prop_types.append('text') # Empty treated as text

            # Map sequence headers strictly and exactly
            header_to_idx = {h: idx for idx, h in enumerate(viewer.full_headers)}
            node_updates = {}
            matched_count = 0
            unmatched_count = 0

            for df_row_idx in range(2, df.shape[0]):
                header_val = df.iloc[df_row_idx, 0]
                if pd.isna(header_val):
                    unmatched_count += 1
                    continue
                header_str = str(header_val).strip()
                
                # Enforce strict exact matching against viewer.full_headers
                if header_str in header_to_idx:
                    node_idx = header_to_idx[header_str]
                    node_updates[node_idx] = df_row_idx
                    matched_count += 1
                else:
                    unmatched_count += 1

            if matched_count == 0:
                raise ValueError("No matching sequence headers found. Enforced strict exact matching against full headers.")

            # Merge properties and types into memory
            for p_idx, prop_name in enumerate(prop_names):
                prop_type = prop_types[p_idx]
                col_idx = valid_cols[p_idx]

                # Check if property already exists
                if prop_name not in viewer.metadata:
                    # Initialize new property
                    if prop_type == 'number':
                        values = np.full(viewer.n_nodes, np.nan, dtype=np.float64)
                    else:
                        values = np.full(viewer.n_nodes, "", dtype=object)
                    viewer.metadata[prop_name] = {
                        "type": prop_type,
                        "values": values
                    }
                else:
                    # Overwrite type and safely convert existing values if type changes
                    old_type = viewer.metadata[prop_name]["type"]
                    viewer.metadata[prop_name]["type"] = prop_type
                    
                    if old_type != prop_type:
                        old_vals = viewer.metadata[prop_name]["values"]
                        if prop_type == 'number':
                            new_vals = np.full(viewer.n_nodes, np.nan, dtype=np.float64)
                            for i in range(viewer.n_nodes):
                                try:
                                    if str(old_vals[i]).strip():
                                        new_vals[i] = float(old_vals[i])
                                except ValueError:
                                    pass
                            viewer.metadata[prop_name]["values"] = new_vals
                        else:
                            new_vals = np.full(viewer.n_nodes, "", dtype=object)
                            for i in range(viewer.n_nodes):
                                if pd.notna(old_vals[i]):
                                    new_vals[i] = str(old_vals[i])
                            viewer.metadata[prop_name]["values"] = new_vals

                # Apply updates from file (ignore empty cells to avoid overwriting with blanks)
                values_arr = viewer.metadata[prop_name]["values"]
                for node_idx, df_row_idx in node_updates.items():
                    cell_val = df.iloc[df_row_idx, col_idx]
                    if pd.isna(cell_val) or str(cell_val).strip() == "" or str(cell_val).strip().lower() == "nan":
                        continue # Skip empty cells to preserve existing viewer metadata
                    
                    if prop_type == 'number':
                        try:
                            values_arr[node_idx] = float(cell_val)
                        except (ValueError, TypeError):
                            pass
                    else:
                        values_arr[node_idx] = str(cell_val)

            successful_files.append(filename)
            matched_nodes.update(node_updates.keys())
            total_unmatched += unmatched_count
            all_merged_props.update(prop_names)

        except Exception as e:
            failed_files.append((filename, str(e)))
            print(f"Error uploading metadata from {filename}: {e}")
            import traceback
            traceback.print_exc()

    # Auto-caching is disabled. Metadata is only held in memory (viewer.metadata) until saved manually.

    # Build response message
    msg_parts = []
    if successful_files:
        msg_parts.append(
            f"Successfully uploaded metadata from {len(successful_files)} file(s): {', '.join(successful_files)}. "
            f"Matched {len(matched_nodes)} unique nodes, ignored {total_unmatched} rows. "
            f"Merged properties: {', '.join(sorted(all_merged_props))}."
        )
        # Inject side panel spreadsheet and refresh its columns
        inject_spreadsheet_panel(viewer)
        if hasattr(viewer, 'metadata_source_model'):
            viewer.metadata_source_model.refresh_columns()
    if failed_files:
        fail_details = "; ".join([f"{f}: {err}" for f, err in failed_files])
        msg_parts.append(f"Failed to upload from {len(failed_files)} file(s): {fail_details}")

    msg = " ".join(msg_parts)
    Command_Engine.print_help(viewer, msg)
