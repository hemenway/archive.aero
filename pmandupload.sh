#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ═══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ROOT="${ROOT:-/Volumes/drive/upload}"
REMOTE="${REMOTE:-r2:charts/sectionals}"
JOBS="${JOBS:-6}"
QUALITY="${QUALITY:-90}"

# rclone tuning
RCLONE_FLAGS=(
  -P
  --s3-upload-concurrency 16
  --s3-chunk-size 128M
  --buffer-size 128M
  --s3-disable-checksum
  --stats 1s
)

# ═══════════════════════════════════════════════════════════════════════════════
#                              TUI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Terminal size
get_term_size() {
  TERM_COLS=$(tput cols 2>/dev/null || echo 80)
  TERM_ROWS=$(tput lines 2>/dev/null || echo 24)
}

# Draw a horizontal line
draw_line() {
  local char="${1:-─}"
  printf '%*s\n' "$TERM_COLS" '' | tr ' ' "$char"
}

# Center text
center_text() {
  local text="$1"
  local width=${#text}
  local padding=$(( (TERM_COLS - width) / 2 ))
  printf "%${padding}s%s\n" '' "$text"
}

# Print header
print_header() {
  clear
  get_term_size
  echo -e "${CYAN}"
  draw_line "═"
  center_text "PMTiles Converter & Uploader"
  draw_line "═"
  echo -e "${NC}"
}

# Print status line
status() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

# Print success
success() {
  echo -e "${GREEN}[✔]${NC} $1"
}

# Print warning
warn() {
  echo -e "${YELLOW}[!]${NC} $1"
}

# Print error
error() {
  echo -e "${RED}[✘]${NC} $1"
}

# ═══════════════════════════════════════════════════════════════════════════════
#                              FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

declare -a ALL_TIFFS=()
declare -a SELECTED_TIFFS=()

discover_files() {
  status "Scanning for TIFF files in $ROOT..."
  mapfile -t ALL_TIFFS < <(find "$ROOT" -type f \( -iname '*.tif' -o -iname '*.tiff' \) -not -path '*/.temp/*' 2>/dev/null | sort)

  if [[ ${#ALL_TIFFS[@]} -eq 0 ]]; then
    warn "No TIFF files found in $ROOT"
    return 1
  fi
  success "Found ${#ALL_TIFFS[@]} TIFF files"
}

# ═══════════════════════════════════════════════════════════════════════════════
#                              PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

process_one() {
  local in="$1"
  local dir base stem mb pm
  dir="$(dirname "$in")"
  base="$(basename "$in")"
  stem="${base%.*}"
  mb="$dir/${stem}.mbtiles"
  pm="$dir/${stem}.pmtiles"

  # Skip if PMTiles already exists
  if [[ -f "$pm" ]]; then
    echo -e "${DIM}⊘ Skipping (pmtiles exists): $base${NC}"
    return 0
  fi

  echo -e "${CYAN}▶${NC} Processing: ${WHITE}$base${NC}"

  # TIFF -> MBTiles
  echo -e "  ${DIM}├─ Converting to MBTiles...${NC}"
  gdal_translate -q -of MBTILES --config GDAL_NUM_THREADS ALL_CPUS -co TILE_FORMAT=WEBP -co QUALITY="$QUALITY" "$in" "$mb"

  # Overviews
  echo -e "  ${DIM}├─ Building overviews...${NC}"
  gdaladdo -q -r bilinear --config GDAL_NUM_THREADS ALL_CPUS "$mb" 2 4 8 16 32 64 128 256

  # MBTiles -> PMTiles
  echo -e "  ${DIM}├─ Converting to PMTiles...${NC}"
  pmtiles convert "$mb" "$pm" 2>/dev/null

  # Remove MBTiles
  rm -f "$mb"

  # Upload PMTiles
  echo -e "  ${DIM}└─ Uploading to R2...${NC}"
  rclone copyto "$pm" "$REMOTE/$(basename "$pm")" "${RCLONE_FLAGS[@]}"

  echo -e "${GREEN}✔${NC} Completed: ${WHITE}$base${NC}"
  echo ""
}

export -f process_one
export ROOT REMOTE QUALITY
export RCLONE_FLAGS
export RED GREEN YELLOW BLUE MAGENTA CYAN WHITE DIM BOLD NC

# ═══════════════════════════════════════════════════════════════════════════════
#                              MENUS
# ═══════════════════════════════════════════════════════════════════════════════

show_config() {
  print_header
  echo -e "${WHITE}Current Configuration:${NC}"
  echo ""
  echo -e "  ${CYAN}Source Directory:${NC}  $ROOT"
  echo -e "  ${CYAN}Remote Target:${NC}     $REMOTE"
  echo -e "  ${CYAN}Parallel Jobs:${NC}     $JOBS"
  echo -e "  ${CYAN}WebP Quality:${NC}      $QUALITY"
  echo ""
  draw_line "─"
  echo ""
}

menu_config() {
  while true; do
    print_header
    echo -e "${WHITE}Configuration${NC}"
    echo ""
    echo -e "  ${CYAN}1)${NC} Source Directory  ${DIM}[$ROOT]${NC}"
    echo -e "  ${CYAN}2)${NC} Remote Target     ${DIM}[$REMOTE]${NC}"
    echo -e "  ${CYAN}3)${NC} Parallel Jobs     ${DIM}[$JOBS]${NC}"
    echo -e "  ${CYAN}4)${NC} WebP Quality      ${DIM}[$QUALITY]${NC}"
    echo ""
    echo -e "  ${CYAN}b)${NC} Back to main menu"
    echo ""
    draw_line "─"
    echo -n "Select option: "

    read -r choice
    case "$choice" in
      1)
        echo -n "Enter source directory: "
        read -r val
        [[ -n "$val" ]] && ROOT="$val"
        ;;
      2)
        echo -n "Enter remote target: "
        read -r val
        [[ -n "$val" ]] && REMOTE="$val"
        ;;
      3)
        echo -n "Enter number of parallel jobs: "
        read -r val
        [[ -n "$val" ]] && JOBS="$val"
        ;;
      4)
        echo -n "Enter WebP quality (1-100): "
        read -r val
        [[ -n "$val" ]] && QUALITY="$val"
        ;;
      b|B) return ;;
      *) warn "Invalid option" ;;
    esac
  done
}

menu_select_files() {
  discover_files || return 1

  # Check for fzf
  if command -v fzf >/dev/null 2>&1; then
    print_header
    echo -e "${WHITE}Select files to process${NC} ${DIM}(TAB to select, ENTER to confirm)${NC}"
    echo ""

    mapfile -t SELECTED_TIFFS < <(printf '%s\n' "${ALL_TIFFS[@]}" | fzf --multi --height=60% --reverse \
      --header="Select TIFF files (TAB=select, ENTER=confirm)" \
      --preview="ls -lh {}" \
      --bind="ctrl-a:select-all,ctrl-d:deselect-all")

    if [[ ${#SELECTED_TIFFS[@]} -eq 0 ]]; then
      warn "No files selected"
      return 1
    fi
    success "Selected ${#SELECTED_TIFFS[@]} files"
  else
    # Fallback: simple numbered list
    print_header
    echo -e "${WHITE}Available TIFF files:${NC}"
    echo ""

    local i=1
    for f in "${ALL_TIFFS[@]}"; do
      local base=$(basename "$f")
      local size=$(ls -lh "$f" 2>/dev/null | awk '{print $5}')
      local pm="${f%.*}.pmtiles"
      local status_icon=""
      [[ -f "$pm" ]] && status_icon="${DIM}[exists]${NC}"
      printf "  ${CYAN}%3d)${NC} %-40s ${DIM}%8s${NC} %s\n" "$i" "$base" "$size" "$status_icon"
      ((i++))
    done

    echo ""
    draw_line "─"
    echo -e "${DIM}Enter file numbers (e.g., 1,3,5-10) or 'all' for all files:${NC}"
    echo -n "> "
    read -r selection

    SELECTED_TIFFS=()
    if [[ "$selection" == "all" ]]; then
      SELECTED_TIFFS=("${ALL_TIFFS[@]}")
    else
      # Parse selection (supports 1,3,5-10 format)
      IFS=',' read -ra parts <<< "$selection"
      for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then
          IFS='-' read -r start end <<< "$part"
          for ((j=start; j<=end; j++)); do
            [[ $j -ge 1 && $j -le ${#ALL_TIFFS[@]} ]] && SELECTED_TIFFS+=("${ALL_TIFFS[$((j-1))]}")
          done
        else
          local idx=$((part))
          [[ $idx -ge 1 && $idx -le ${#ALL_TIFFS[@]} ]] && SELECTED_TIFFS+=("${ALL_TIFFS[$((idx-1))]}")
        fi
      done
    fi

    if [[ ${#SELECTED_TIFFS[@]} -eq 0 ]]; then
      warn "No valid files selected"
      return 1
    fi
    success "Selected ${#SELECTED_TIFFS[@]} files"
  fi

  sleep 1
  return 0
}

menu_process() {
  if [[ ${#SELECTED_TIFFS[@]} -eq 0 ]]; then
    # Auto-select all if none selected
    discover_files || return 1
    SELECTED_TIFFS=("${ALL_TIFFS[@]}")
  fi

  print_header
  show_config

  echo -e "${WHITE}Files to process: ${CYAN}${#SELECTED_TIFFS[@]}${NC}"
  echo ""

  # Show preview of files
  local count=0
  for f in "${SELECTED_TIFFS[@]}"; do
    local base=$(basename "$f")
    local pm="${f%.*}.pmtiles"
    if [[ -f "$pm" ]]; then
      echo -e "  ${DIM}⊘ $base (will skip - pmtiles exists)${NC}"
    else
      echo -e "  ${CYAN}▶${NC} $base"
    fi
    ((count++))
    [[ $count -ge 10 ]] && { echo -e "  ${DIM}... and $((${#SELECTED_TIFFS[@]} - 10)) more${NC}"; break; }
  done

  echo ""
  draw_line "─"
  echo -e "${YELLOW}Start processing? (y/n)${NC} "
  read -r confirm

  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    warn "Cancelled"
    sleep 1
    return 1
  fi

  echo ""
  status "Starting processing with $JOBS parallel jobs..."
  echo ""
  draw_line "─"
  echo ""

  # Process files
  printf '%s\n' "${SELECTED_TIFFS[@]}" | xargs -n 1 -P "$JOBS" bash -lc 'process_one "$0"'

  echo ""
  draw_line "─"
  success "Processing complete!"
  echo ""
  echo -e "${DIM}Press Enter to continue...${NC}"
  read -r
}

menu_status() {
  print_header
  echo -e "${WHITE}Status Overview${NC}"
  echo ""

  discover_files 2>/dev/null || { echo "No files found"; sleep 2; return; }

  local total=${#ALL_TIFFS[@]}
  local completed=0
  local pending=0
  local total_size=0
  local completed_size=0

  for f in "${ALL_TIFFS[@]}"; do
    local pm="${f%.*}.pmtiles"
    local size=$(stat -f%z "$f" 2>/dev/null || echo 0)
    total_size=$((total_size + size))

    if [[ -f "$pm" ]]; then
      ((completed++))
      completed_size=$((completed_size + size))
    else
      ((pending++))
    fi
  done

  # Format sizes
  local total_gb=$(echo "scale=2; $total_size / 1073741824" | bc)
  local completed_gb=$(echo "scale=2; $completed_size / 1073741824" | bc)
  local pct=0
  [[ $total -gt 0 ]] && pct=$((completed * 100 / total))

  echo -e "  ${CYAN}Total TIFF files:${NC}     $total"
  echo -e "  ${GREEN}Completed:${NC}            $completed"
  echo -e "  ${YELLOW}Pending:${NC}              $pending"
  echo ""
  echo -e "  ${CYAN}Total size:${NC}           ${total_gb} GB"
  echo -e "  ${GREEN}Processed:${NC}            ${completed_gb} GB"
  echo ""

  # Progress bar
  local bar_width=40
  local filled=$((pct * bar_width / 100))
  local empty=$((bar_width - filled))

  printf "  Progress: ["
  printf "${GREEN}%${filled}s${NC}" '' | tr ' ' '█'
  printf "${DIM}%${empty}s${NC}" '' | tr ' ' '░'
  printf "] %d%%\n" "$pct"

  echo ""
  draw_line "─"
  echo ""
  echo -e "${DIM}Press Enter to continue...${NC}"
  read -r
}

menu_main() {
  while true; do
    print_header
    echo -e "${WHITE}Main Menu${NC}"
    echo ""
    echo -e "  ${CYAN}1)${NC} Process all files"
    echo -e "  ${CYAN}2)${NC} Select specific files"
    echo -e "  ${CYAN}3)${NC} View status"
    echo -e "  ${CYAN}4)${NC} Configuration"
    echo ""
    echo -e "  ${CYAN}q)${NC} Quit"
    echo ""
    draw_line "─"
    echo -n "Select option: "

    read -r choice
    case "$choice" in
      1)
        SELECTED_TIFFS=()
        menu_process
        ;;
      2)
        menu_select_files && menu_process
        ;;
      3)
        menu_status
        ;;
      4)
        menu_config
        ;;
      q|Q)
        echo ""
        success "Goodbye!"
        exit 0
        ;;
      *)
        warn "Invalid option"
        sleep 0.5
        ;;
    esac
  done
}

# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# Check dependencies
check_deps() {
  local missing=()
  for cmd in gdal_translate gdaladdo pmtiles rclone; do
    command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    error "Missing required commands: ${missing[*]}"
    exit 127
  fi
}

# Parse arguments
show_help() {
  cat << EOF
PMTiles Converter & Uploader

Usage: $(basename "$0") [OPTIONS]

Options:
  -h, --help          Show this help message
  -b, --batch         Run in batch mode (no TUI, process all)
  -r, --root DIR      Source directory (default: $ROOT)
  -t, --target REMOTE Remote target (default: $REMOTE)
  -j, --jobs N        Parallel jobs (default: $JOBS)
  -q, --quality N     WebP quality 1-100 (default: $QUALITY)

Examples:
  $(basename "$0")                    # Interactive TUI
  $(basename "$0") -b                 # Batch process all
  $(basename "$0") -b -j 4 -q 85      # Batch with options
EOF
}

BATCH_MODE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) show_help; exit 0 ;;
    -b|--batch) BATCH_MODE=true; shift ;;
    -r|--root) ROOT="$2"; shift 2 ;;
    -t|--target) REMOTE="$2"; shift 2 ;;
    -j|--jobs) JOBS="$2"; shift 2 ;;
    -q|--quality) QUALITY="$2"; shift 2 ;;
    *) error "Unknown option: $1"; show_help; exit 1 ;;
  esac
done

check_deps

if $BATCH_MODE; then
  # Batch mode - original behavior
  status "Running in batch mode..."
  find "$ROOT" -type f \( -iname '*.tif' -o -iname '*.tiff' \) -not -path '*/.temp/*' \
  | sort \
  | xargs -n 1 -P "$JOBS" bash -lc 'process_one "$0"'
else
  # Interactive TUI
  menu_main
fi
