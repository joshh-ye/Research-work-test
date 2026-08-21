#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Per-baseline comparison figures for the four-way benchmark, in R/ggplot2.
#
# The Python side (compare_transfer_enformer.py) produces this figure family for
# Enformer only. This script produces the same family for ALL THREE baselines —
# pretrained Borzoi, Enformer and AlphaGenome — from the single reduced table
# results_benchmark/benchmark_comparison_{split}.csv, so every panel is drawn
# from the same numbers on the same shared 896 x 128 bp grid.
#
# Six figures per baseline:
#   violin        distribution of per-track R, transfer vs baseline
#   ecdf          cumulative distribution of both
#   scatter       per-target transfer vs baseline, with y = x
#   delta_hist    histogram of the paired difference
#   paired        per-target dumbbell, sorted by baseline R
#   plausibility  is the matched track the same TF, a relative, or unrelated?
#
# On ACCRE:
#   module load r
#   export R_LIBS_USER=~/R/x86_64-pc-linux-gnu-library/4.6
#   Rscript plot_benchmark_comparison.R
# In RStudio Server: open the project and source() this file.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  need <- c("ggplot2", "dplyr", "tidyr", "scales")
  miss <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
  if (length(miss)) {
    stop("Missing R packages: ", paste(miss, collapse = ", "), "\n",
         "  install.packages(c(", paste0('"', miss, '"', collapse = ", "), "))",
         call. = FALSE)
  }
  library(ggplot2); library(dplyr); library(tidyr); library(scales)
})

# --- config ----------------------------------------------------------------
args     <- commandArgs(trailingOnly = TRUE)
split    <- if (length(args) >= 1) args[1] else "test"
in_csv   <- file.path("results_benchmark", sprintf("benchmark_comparison_%s.csv", split))
out_dir  <- "results_comparison_R"
dir.create(out_dir, showWarnings = FALSE)

# validated categorical palette (same hues as the Python figures, so the two
# sets can sit side by side in one deck without reading as different studies)
C_TL <- "#2a78d6"; C_BZ <- "#eb6834"; C_AG <- "#1baf7a"; C_EF <- "#4a3aa7"
INK  <- "#0b0b0b"; INK2 <- "#52514e"; GRID <- "#d9d8d4"; SURF <- "#fcfcfb"

BASELINES <- list(
  list(key = "borzoi_pretrained", label = "Borzoi pretrained", colour = C_BZ),
  list(key = "enformer",          label = "Enformer",          colour = C_EF),
  list(key = "alphagenome",       label = "AlphaGenome",       colour = C_AG)
)

theme_bench <- function() {
  theme_minimal(base_size = 11) +
    theme(
      plot.background   = element_rect(fill = SURF, colour = NA),
      panel.background  = element_rect(fill = SURF, colour = NA),
      panel.grid.minor  = element_blank(),
      panel.grid.major  = element_line(colour = GRID, linewidth = 0.3),
      axis.text         = element_text(colour = INK2),
      axis.title        = element_text(colour = INK),
      plot.title        = element_text(colour = INK, face = "bold", size = 12.5),
      plot.subtitle     = element_text(colour = INK2, size = 9.5),
      legend.position   = "top",
      legend.title      = element_blank(),
      legend.text       = element_text(colour = INK)
    )
}

save_both <- function(p, name, w = 7.2, h = 4.8) {
  for (ext in c("png", "pdf")) {
    ggsave(file.path(out_dir, paste0(name, "_", split, ".", ext)),
           p, width = w, height = h, dpi = 200, bg = SURF)
  }
}

# TF name out of a matched-track description. Borzoi and Enformer use
# "CHIP:<factor>:<cell/tissue>"; AlphaGenome uses "<factor> (<biosample>)".
tf_of <- function(x, key) {
  if (key == "alphagenome") toupper(trimws(sub("\\s*\\(.*$", "", x)))
  else                      toupper(trimws(vapply(strsplit(x, ":"), function(p)
                              if (length(p) >= 2) p[2] else "", character(1))))
}

# Crude TF-family grouping, mirroring borzoi_baseline_utils._family so the R and
# Python plausibility panels agree.
FAMS <- c("FOX","GATA","HOX","SOX","KLF","STAT","IRF","SMAD","TCF","LEF","NFK",
          "NR","ZNF","E2F","ETS","ELF","CEBP","JUN","FOS","MYC","RUNX","TEAD","TP","SP")
family_of <- function(f) {
  f <- toupper(f)
  hit <- FAMS[vapply(FAMS, function(p) startsWith(f, p), logical(1))]
  if (length(hit)) hit[which.max(nchar(hit))] else substr(f, 1, 3)
}

classify <- function(target_tf, match_tf) {
  if (is.na(match_tf) || match_tf == "" || match_tf == "NAN") return("unclear")
  if (target_tf != "" && target_tf == match_tf)               return("same TF")
  if (target_tf != "" && family_of(target_tf) == family_of(match_tf)) return("related family")
  "different TF"
}

# --- data ------------------------------------------------------------------
if (!file.exists(in_csv)) stop("missing ", in_csv, " — run benchmark_compare_all.py first")
d <- read.csv(in_csv, stringsAsFactors = FALSE)
cat(sprintf("Loaded %d targets from %s\n", nrow(d), in_csv))

summary_rows <- list()

for (b in BASELINES) {
  key <- b$key; lab <- b$label; col <- b$colour
  rcol <- paste0(key, "_r"); tcol <- paste0(key, "_track")

  df <- data.frame(
    target      = d$target_factor,
    cell        = d$target_cell_line,
    transfer_r  = d$borzoi_tl_r,
    baseline_r  = d[[rcol]],
    match_track = d[[tcol]],
    stringsAsFactors = FALSE
  )
  df$delta  <- df$transfer_r - df$baseline_r
  df$winner <- ifelse(abs(df$delta) <= 0.01, "tie",
                      ifelse(df$delta > 0, "Borzoi-TL", lab))

  pal <- c("Borzoi-TL" = C_TL); pal[lab] <- col

  # 1. violin + box ---------------------------------------------------------
  long <- df |>
    select(transfer_r, baseline_r) |>
    pivot_longer(everything(), names_to = "model", values_to = "r") |>
    mutate(model = factor(ifelse(model == "transfer_r", "Borzoi-TL", lab),
                          levels = c("Borzoi-TL", lab)))
  means <- long |> group_by(model) |> summarise(m = mean(r), .groups = "drop")

  p1 <- ggplot(long, aes(model, r, fill = model, colour = model)) +
    geom_violin(alpha = 0.18, linewidth = 0.7, width = 0.85) +
    geom_boxplot(width = 0.16, outlier.shape = NA, fill = "white", linewidth = 0.7) +
    # pinned above the violin, not at the mean, so it never covers the box
    geom_text(data = means, aes(model, 0.985, label = sprintf("%.3f", m)),
              vjust = 1, fontface = "bold", size = 4.0, show.legend = FALSE) +
    scale_fill_manual(values = pal) + scale_colour_manual(values = pal) +
    labs(title = sprintf("Per-track accuracy: Borzoi-TL vs %s", lab),
         subtitle = sprintf("%d held-out CRC TF ChIP-seq tracks · shared 896 x 128 bp grid · bold = mean", nrow(df)),
         x = NULL, y = "Pearson R per target track") +
    ylim(0, 1.02) + theme_bench() + theme(legend.position = "none")
  save_both(p1, paste0(key, "_violin"))

  # 2. ECDF -----------------------------------------------------------------
  p2 <- ggplot(long, aes(r, colour = model)) +
    stat_ecdf(linewidth = 1.1) +
    scale_colour_manual(values = pal) +
    labs(title = sprintf("Cumulative distribution of per-track R"),
         subtitle = sprintf("a curve further right is better · Borzoi-TL vs %s", lab),
         x = "Pearson R", y = "fraction of tracks at or below") +
    theme_bench()
  save_both(p2, paste0(key, "_ecdf"))

  # 3. scatter --------------------------------------------------------------
  p3 <- ggplot(df, aes(baseline_r, transfer_r)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = INK2, linewidth = 0.5) +
    geom_point(aes(colour = winner), size = 2.1, alpha = 0.8) +
    scale_colour_manual(values = c(pal, tie = INK2)) +
    coord_equal(xlim = c(0.1, 0.95), ylim = c(0.1, 0.95)) +
    labs(title = sprintf("Per-target comparison vs %s", lab),
         subtitle = "above the dashed line = Borzoi-TL better",
         x = sprintf("%s best-match R", lab), y = "Borzoi-TL R") +
    theme_bench()
  save_both(p3, paste0(key, "_scatter"), w = 6.0, h = 6.0)

  # 4. paired-difference histogram -----------------------------------------
  mu <- mean(df$delta)
  p4 <- ggplot(df, aes(delta)) +
    geom_histogram(bins = 34, fill = col, colour = SURF, alpha = 0.85, linewidth = 0.3) +
    geom_vline(xintercept = 0, colour = INK2, linewidth = 0.6) +
    geom_vline(xintercept = mu, colour = C_TL, linewidth = 1.0, linetype = "dashed") +
    annotate("text", x = mu, y = Inf, vjust = 1.8, hjust = -0.08,
             label = sprintf("mean %+.3f", mu), colour = C_TL, fontface = "bold", size = 3.6) +
    labs(title = sprintf("Paired difference vs %s", lab),
         subtitle = sprintf("%d Borzoi-TL wins · %d losses · %d ties (|ΔR| ≤ 0.01)",
                            sum(df$winner == "Borzoi-TL"), sum(df$winner == lab), sum(df$winner == "tie")),
         x = "ΔR  (Borzoi-TL minus baseline)", y = "number of target tracks") +
    theme_bench()
  save_both(p4, paste0(key, "_delta_hist"))

  # 5. paired dumbbell ------------------------------------------------------
  ord <- df |> arrange(baseline_r) |> mutate(i = row_number())
  p5 <- ggplot(ord) +
    geom_segment(aes(x = i, xend = i, y = baseline_r, yend = transfer_r),
                 colour = GRID, linewidth = 0.4) +
    geom_point(aes(i, baseline_r, colour = lab), size = 0.9) +
    geom_point(aes(i, transfer_r, colour = "Borzoi-TL"), size = 0.9) +
    scale_colour_manual(values = pal) +
    labs(title = sprintf("Every target, sorted by %s performance", lab),
         subtitle = "each vertical stem links one target's two scores",
         x = "target track (sorted)", y = "Pearson R") +
    theme_bench()
  save_both(p5, paste0(key, "_paired"), w = 8.0, h = 4.4)

  # 6. biological plausibility ---------------------------------------------
  mtf <- tf_of(df$match_track, key)
  df$plaus <- vapply(seq_len(nrow(df)),
                     function(i) classify(toupper(trimws(df$target[i])), mtf[i]), character(1))
  pl <- df |> count(plaus) |> mutate(plaus = reorder(plaus, n))
  p6 <- ggplot(pl, aes(n, plaus)) +
    geom_col(fill = col, alpha = 0.85, width = 0.62) +
    geom_text(aes(label = n), hjust = -0.35, colour = INK, fontface = "bold", size = 3.6) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.14))) +
    labs(title = sprintf("Is the matched %s track the same factor?", lab),
         subtitle = "best-match is selected purely by correlation — TF identity is not constrained",
         x = "number of target tracks", y = NULL) +
    theme_bench()
  save_both(p6, paste0(key, "_plausibility"), w = 6.6, h = 3.6)

  summary_rows[[key]] <- data.frame(
    baseline = lab, n = nrow(df),
    transfer_mean = mean(df$transfer_r), baseline_mean = mean(df$baseline_r),
    mean_delta = mu, median_delta = median(df$delta),
    wins = sum(df$winner == "Borzoi-TL"), losses = sum(df$winner == lab),
    ties = sum(df$winner == "tie"),
    wilcoxon_p = suppressWarnings(wilcox.test(df$transfer_r, df$baseline_r, paired = TRUE)$p.value),
    same_TF_matches = sum(df$plaus == "same TF")
  )
  cat(sprintf("  %-18s mean Δ=%+.4f  %d/%d/%d W/L/T  -> 6 figures\n",
              lab, mu, sum(df$winner == "Borzoi-TL"), sum(df$winner == lab), sum(df$winner == "tie")))
}

summ <- bind_rows(summary_rows)
write.csv(summ, file.path(out_dir, sprintf("summary_by_baseline_%s.csv", split)), row.names = FALSE)
print(summ, digits = 4)
cat(sprintf("\nWrote %d figures (PNG+PDF) and 1 table to %s/\n", 6 * length(BASELINES), out_dir))
