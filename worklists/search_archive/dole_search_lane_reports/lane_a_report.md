# LANE A report — round 5: commercial aviation chart CD-ROM products (~1994-2010) as ISOs/rips

Verified findings: **1** (low value — see caveat). Main output of this lane is a set of
hard closures on the CD-ROM-product front plus two named record-only leads.

## Positives

1. **PC Pilot Issue 82 cover CD (IA `cdrinc_PC_Pilot_Issue_82`)** — `Flight Adventure/CD
   Intro/Ketchikan Sectional Chart.jpg`: full-sheet Ketchikan sectional WITH legend,
   10924x8207 px, 26.3 MB RGB JPG, file dated 2012-07-31 → edition 52 (2012-04-05).
   Extractable via IA on-the-fly view_archive URL (302→200, full 26MB download verified).
   CAVEAT: dole already holds ed 52 (wayback `Ketchikan_52.zip`); almost certainly a
   rendition of the free FAA raster, not a distinct scan. Recorded in CSV with caveat.
   Mirror of same disc: IA item `pc_pilot_-_issue_82_uk`. Scanned ALL other PC Pilot CDs
   on IA (issues 1-21, 29, 30, 31, 38, 45, 47, 83, 86, 101, 102, 103, 105) — no other
   full sectional; their chart folders are IFR approach plates / UK-NZ terminal charts
   (excluded classes). Issue 82 is the only "Flight Adventure" disc with a chart.

2. **NOAA Aeronautical Data Sampler CD-ROM Dec 1988 (IA item, 50MB ISO) — now CLOSED.**
   Downloaded and inspected (High Sierra format, volume "NOAA", Meridian Data CD Publisher).
   Content = "aeronautical data file sampler" produced with U. Maryland Research
   Foundation: Turbo-C viewer + attribute/data files. ZERO raster images (no .TIF/.PCX/
   .BMP/.GIF anywhere in the image). Confirms the Part-5 inference from the 1990
   Sampler II — the Part-3 "unverified candidate" can be struck.

3. **MapTech aviation product line identified (record-only lead).** Wayback maptech.com:
   `/air/` subtree (2001-2007) sold **"VFR/IFR AeroPacks"** (regional CD-ROMs of scanned
   FAA charts in BSB format, the aviation sibling of their marine Digital Charts) plus
   "VFR/IFR AeroSims" (flight-sim editions) and "Chart Navigator/AeroViewer" free viewer.
   Free demo `ftp://ftp.maptech.com/downloads/AERO_DEMO.EXE` (23.8MB) contained Chart
   Navigator + one Boston **TAC** (excluded class) — and no ftp.maptech.com captures exist
   in Wayback anyway (3 urlkeys, all 404). No AeroPack rips on IA/discmaster/redump/
   winworldpc/macintoshgarden; websearch nil. **AeroPack CDs (~1999-2006, sectionals in BSB
   v3) are the single most promising physical-CD target for the NACO-era gap** — used-market
   discs are the only known survival. Future round: hunt "Maptech AeroPack" / "AeroSim"
   on used-software marketplaces, marine-BSB hoarder collections (the IA `NOAA_RNC` 6GB
   BSB dump shows such hoards exist for marine), and Sporty's-catalog-era pilot forums.

4. **Northstar CT-1000 "Digital Flight Bag" drive dump (IA `northstar_202511`, 106MB) —
   inspected, closed.** Win98 drive image: Jeppesen **JeppView** install with vector .BFD
   chart databases (IFR terminal charts, excluded class). No sectional rasters.

## Zero-yield (all verified, log-style)

archive.org advancedsearch, product-name queries (all 0 relevant unless noted):
"maptech"+aviation/sectional/chartkit, "pocket sectional(s)", "chartkit", "terrain
navigator"+aviation, "flitesoft", "RMS Technology" (noise), "flitestar" (→ MentorPlus
FliteStar 2.0, 1.9MB 7z, vector planner, no rasters), "jeppview" (manual only),
"navsuite", jeppesen+software (16 items: FlitePro 6.2 x2, SIMCharts Europe, Dataplan
JeppLink — all vector/IFR/Europe), "anywhere map"/"anywheremap"/"control vision" (manuals
+ Android apk, vector product), "rasterplus"/"vfr rasterplus", sporty's/sportys variants,
"seattle avionics" (Voyager manuals only), "flightprep"/"chartcase", "echo flight",
"flitemap", "destination direct", delorme+aviation, "memory-map"+aviation, "simcharts"
(Jeppesen approach charts = excluded), "mountainscope", "pcavionics", "fugawi" (topo/
marine), "chart navigator" (manuals), "teletype"+gps (vector aviation maps), "chartdata",
"golden eagle"+flightprep, "sectionals" (sports noise), "vfr sectional", "naco"+charts
(NACOmatic = IFR plates), "air chart systems", "howie keefe", "aeropack", "aerosim",
maptech+bsb (only marine NOAA_RNC), "bsb"+aeronautical, "mentorplus"/"mentor plus"
(+ CIRRUS DUATS ISO = briefing client), aopa/asa/"flight guide"/dauntless/"qct"+charts,
title:vfr|faa|noaa mediatype:software (only known items), collection:cdromsoftware/
cd-roms aviation browses (flight-sim shovelware; Aviation 2000, WinPlanner Plus, BAO
Flight Shop, Pro Pilot USA, FRFPVRFO all inspected via metadata/listings — sim content).

ISO/zip inspections (contents verified, no sectional rasters):
- `jeppesen-flitepro-6.2.0` FLITEPRO62.iso (365MB): InstallShield cabs, IFR sim, no rasters.
- `FRFPVRFO` (81MB, raw Mode2/XA bin): "ATC Radio" = AUSTRALIAN radio-procedures trainer
  (ToolBook + WAVs), zero charts.
- `northstar_202511`, NOAA 1988 sampler: above.
- `Simcharts` (17MB zip): Jeppesen SIMCharts = approach charts, excluded.

discmaster.textfiles.com (1.7B files indexed inside IA CD/FTP images) — strong negative
for the whole textfiles CD corpus: filename searches "sectional" (200 hits = furniture/
Half-Life/anatomy noise; sole aviation hit = the PC Pilot 82 Ketchikan JPG), "aeronautical"
(9, none charts), "flitesoft", "rasterplus", "chartkit", "flitestar", "jeppview", "naco"
(83, none aviation), "aeronav", "seclosa", "secsea": 0. No NACO/Jeppesen/MapTech chart CD
has been ingested there.

Abandonware/disc DBs: winworldpc (jeppesen: none), macintoshgarden ("sectional chart",
"aeronautical": none), redump.org quicksearch (jeppesen/maptech/flitestar/sectional/
aeronautical: no discs), vetusware (ECONNRESET bot-wall), myabandonware (browser-check wall).

Wayback CDX vendor-domain sweeps (new hosts this round, none previously in log):
- rmstek.com (RMS Technology, 1,187 urlkeys): FliteSoft chart products are **vector**
  ("ChartMaster CD" = vector map per 2000 product page; raster.htm is a generic explainer).
  No chart-data binaries captured. Kills the "RasterPlus" product-name theory.
- flightprep.com (22,122): ChartCase 2007 free download = program + VECTOR nav data
  (182MB, forward.php links); raster charts were ChartKey-subscription, never captured.
  iChart page JPGs are excerpts (excluded).
- seattleavionics.com (1,952): ChartData was an authenticated download portal
  (Default.aspx?TargetDevice=...); only na.air.zip 17MB vector data captured. No rasters.
- controlvision.com / anywheremap.com (52k): vector product, nothing.
- echoflight.com (525): EchoChart pages, .wmv demo only.
- teletype.com (14,660): aviation edition = vector maps (screenshots prove); chart CDs
  captured are marine/bathymetric.
- memory-map.com (2,099): QuickCharts QC* = NOAA **marine**; US aviation QCT product not
  evidenced; only mmnav program exes captured.
- dtint.com (Destination Direct vendor domain, 8,958): unrelated corporate site content.
- mentorplus.com (1,923): brochure pages only.
- maptech.com: /air/ subtree as in Positives #3; support/downloads has NO binaries
  captured (all .cfm HTML; download host was ftp.maptech.com, uncaptured).
CONCLUSION extending log Part-4: not just NACO — ALL third-party raster-chart vendors of
1999-2010 (MapTech, FlightPrep, Seattle Avionics, EchoFlight, TeleType) distributed
rasters on subscription CD/authenticated download, and none of it was web-crawled.

Other: Aviation Consumer 2001 "Computer Flight Planners" review confirms Big Three
(FliteStar/FliteSoft/Destination Direct) were all vector — chart-CD hunting for those
titles is pointless. IA uploader enumerations: jaj86@hotmail.com (143 items, flight-sim
CD archiver), renatotome@aol.com (generic Brazilian CD dumps), dewyattmail (FlitePro
only), cdrinc_* collection (900 idents paged; only PC Pilot + Flightline's ADs + general
map CDs). websearch "NACO/NOAA sectional raster CD ISO": nil.

## Record-only / leads for future rounds
- **MapTech VFR/IFR AeroPack CDs** — top target, see Positives #3.
- NACO "Sectional Raster Chart" subscription CDs — still zero rips located anywhere;
  physical media hunt only.
- Jeppesen FliteStar/FliteMap "RasterPack"-era chart CDs: no evidence such a raster
  product existed for sectionals (JeppView = IFR); deprioritize.
- vetusware.com + myabandonware.com remain unenumerable by tooling (bot-walls) — add to
  the manual-browser list; both are plausible ISOs hosts for AeroPack/FliteSoft discs.
- PC Pilot cover CDs not on IA (issues 22-28, 32-44, 48-81, 84+) — issue-82-style full
  sectionals may recur in other "Flight Adventure" issues (2010-2015, low era value).
