# chord.R — R interface to CHORD via reticulate
# Usage:
#   source("chord.R")
#   chord_setup()
#   results <- chord_run(expr_matrix, timepoints)

# ===========================================================================
# Setup
# ===========================================================================

#' Initialise the CHORD Python environment.
#'
#' @param python_path Path to the Python binary. If NULL, auto-detects from
#'   the conda environment "omicverse2".
#' @param chord_src   Path to the CHORD source tree (the directory that
#'   contains the `chord` package). If NULL, assumes the package is already
#'   importable.
#' @return invisible(TRUE)
chord_setup <- function(python_path = NULL, chord_src = NULL) {
  if (!requireNamespace("reticulate", quietly = TRUE))
    stop("Package 'reticulate' is required. Install it with install.packages('reticulate').")

  # Auto-detect python from conda env 'omicverse2' --------------------------
  if (is.null(python_path)) {
    tryCatch({
      reticulate::use_condaenv("omicverse2", required = TRUE)
    }, error = function(e) {
      message("Could not activate conda env 'omicverse2': ", conditionMessage(e))
      message("Falling back to default Python.")
    })
  } else {
    reticulate::use_python(python_path, required = TRUE)
  }

  # Optionally add chord source to sys.path ---------------------------------
  if (!is.null(chord_src)) {
    chord_src <- normalizePath(chord_src, mustWork = TRUE)
    sys <- reticulate::import("sys")
    if (!(chord_src %in% sys$path)) sys$path$insert(0L, chord_src)
  }

  # Import chord and stash in a package-level env ---------------------------
  tryCatch({
    .chord_env$chord <- reticulate::import("chord")
    .chord_env$np    <- reticulate::import("numpy")
    .chord_env$pd    <- reticulate::import("pandas")
  }, error = function(e) {
    stop("Failed to import CHORD Python package: ", conditionMessage(e),
         "\nMake sure CHORD is installed or pass chord_src= to chord_setup().")
  })

  invisible(TRUE)
}

# Private environment to cache imported modules
.chord_env <- new.env(parent = emptyenv())

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

.check_setup <- function() {
  if (is.null(.chord_env$chord))
    stop("CHORD is not initialised. Call chord_setup() first.")
}

#' Convert an R matrix (genes x timepoints) to a numpy array.
.to_numpy <- function(mat) {
  if (!is.matrix(mat)) mat <- as.matrix(mat)
  .chord_env$np$array(mat, dtype = "float64")
}

#' Convert a pandas DataFrame to an R data.frame.
.pd_to_r <- function(pdf) {
  tryCatch(
    reticulate::py_to_r(pdf),
    error = function(e) {
      # Fallback: manual conversion via CSV round-trip
      tmp <- tempfile(fileext = ".csv")
      on.exit(unlink(tmp), add = TRUE)
      pdf$to_csv(tmp, index = TRUE)
      utils::read.csv(tmp, row.names = 1, check.names = FALSE)
    }
  )
}

# ===========================================================================
# Core detection functions
# ===========================================================================

#' Run BHDT rhythmicity detection.
#'
#' @param expr_matrix Numeric matrix (genes x timepoints) with gene names as
#'   rownames.
#' @param timepoints  Numeric vector of sampling times (hours).
#' @param method      BHDT method: "analytic" (default) or "mcmc".
#' @param n_jobs      Number of parallel jobs (-1 = all cores).
#' @param verbose     Logical; print progress?
#' @return A data.frame with BHDT results (one row per gene).
chord_detect <- function(expr_matrix, timepoints, method = "analytic",
                         n_jobs = -1L, verbose = TRUE) {
  .check_setup()
  stopifnot(is.numeric(timepoints), length(timepoints) == ncol(expr_matrix))

  np_expr <- .to_numpy(expr_matrix)
  np_tp   <- .chord_env$np$array(as.numeric(timepoints), dtype = "float64")

  tryCatch({
    res <- .chord_env$chord$detect(
      np_expr,
      np_tp,
      method  = method,
      n_jobs  = as.integer(n_jobs),
      verbose = verbose
    )
    out <- .pd_to_r(res)
    if (!is.null(rownames(expr_matrix)) && nrow(out) == nrow(expr_matrix))
      rownames(out) <- rownames(expr_matrix)
    out
  }, error = function(e) {
    stop("chord_detect failed: ", conditionMessage(e))
  })
}

#' Run PINOD oscillator decomposition.
#'
#' @param expr_matrix Numeric matrix (genes x timepoints).
#' @param timepoints  Numeric vector of sampling times (hours).
#' @param n_epochs    Number of training epochs (default 300).
#' @param device      Torch device string ("cpu" or "cuda").
#' @param verbose     Logical; print progress?
#' @return A data.frame with oscillator parameters.
chord_decompose <- function(expr_matrix, timepoints, n_epochs = 300L,
                            device = "cpu", verbose = TRUE) {
  .check_setup()
  stopifnot(is.numeric(timepoints), length(timepoints) == ncol(expr_matrix))

  np_expr <- .to_numpy(expr_matrix)
  np_tp   <- .chord_env$np$array(as.numeric(timepoints), dtype = "float64")

  tryCatch({
    res <- .chord_env$chord$decompose(
      np_expr,
      np_tp,
      n_epochs = as.integer(n_epochs),
      device   = device,
      verbose  = verbose
    )
    out <- .pd_to_r(res)
    if (!is.null(rownames(expr_matrix)) && nrow(out) == nrow(expr_matrix))
      rownames(out) <- rownames(expr_matrix)
    out
  }, error = function(e) {
    stop("chord_decompose failed: ", conditionMessage(e))
  })
}

#' Run the full CHORD pipeline (detection + optional decomposition).
#'
#' @param expr_matrix Numeric matrix (genes x timepoints).
#' @param timepoints  Numeric vector of sampling times (hours).
#' @param methods     Character: "auto", "bhdt", "pinod", or "both".
#' @param bhdt_method BHDT method ("analytic" or "mcmc").
#' @param n_jobs      Number of parallel jobs.
#' @param verbose     Logical; print progress?
#' @return A data.frame with combined results.
chord_run <- function(expr_matrix, timepoints, methods = "auto",
                      bhdt_method = "analytic", n_jobs = -1L,
                      verbose = TRUE) {
  .check_setup()
  stopifnot(is.numeric(timepoints), length(timepoints) == ncol(expr_matrix))

  np_expr <- .to_numpy(expr_matrix)
  np_tp   <- .chord_env$np$array(as.numeric(timepoints), dtype = "float64")

  tryCatch({
    res <- .chord_env$chord$run(
      np_expr,
      np_tp,
      methods     = methods,
      bhdt_method = bhdt_method,
      n_jobs      = as.integer(n_jobs),
      verbose     = verbose
    )
    out <- .pd_to_r(res)
    if (!is.null(rownames(expr_matrix)) && nrow(out) == nrow(expr_matrix))
      rownames(out) <- rownames(expr_matrix)
    out
  }, error = function(e) {
    stop("chord_run failed: ", conditionMessage(e))
  })
}

# ===========================================================================
# Utility functions
# ===========================================================================

#' Return the CHORD version string.
chord_version <- function() {
  .check_setup()
  tryCatch(
    as.character(.chord_env$chord$`__version__`),
    error = function(e) "unknown"
  )
}

#' Print a classification summary table from CHORD results.
#'
#' @param results A data.frame returned by chord_detect / chord_run.
#'   Must contain a "classification" column.
chord_classify_summary <- function(results) {
  if (!"classification" %in% colnames(results))
    stop("results must contain a 'classification' column.")
  tbl <- table(results$classification)
  cat("CHORD classification summary\n")
  cat(strrep("-", 40), "\n")
  for (nm in names(tbl))
    cat(sprintf("  %-25s %d\n", nm, tbl[[nm]]))
  cat(strrep("-", 40), "\n")
  cat(sprintf("  %-25s %d\n", "Total", sum(tbl)))
  invisible(tbl)
}

#' Quick plot of a single gene's expression with its fitted curve.
#'
#' @param expr_vector Numeric vector of expression values across timepoints.
#' @param timepoints  Numeric vector of sampling times (hours).
#' @param result_row  A single-row data.frame (or named list) from CHORD
#'   results, expected to contain "amplitude", "phase", and "period" fields.
chord_plot_gene <- function(expr_vector, timepoints, result_row) {
  stopifnot(length(expr_vector) == length(timepoints))

  amp    <- as.numeric(result_row[["amplitude"]])
  phase  <- as.numeric(result_row[["phase"]])
  period <- as.numeric(result_row[["period"]])
  mesor  <- mean(expr_vector, na.rm = TRUE)

  gene_name <- if (!is.null(result_row[["gene"]])) result_row[["gene"]]
               else if (!is.null(names(expr_vector))) names(expr_vector)[1]
               else "gene"

  t_fine <- seq(min(timepoints), max(timepoints), length.out = 200)
  y_fit  <- mesor + amp * cos(2 * pi / period * t_fine - phase)

  plot(timepoints, expr_vector,
       pch = 16, col = "steelblue",
       xlab = "Time (hours)", ylab = "Expression",
       main = paste0(gene_name, "  (T=", round(period, 1), " h)"))
  lines(t_fine, y_fit, col = "tomato", lwd = 2)
  legend("topright",
         legend = c("observed", "fitted"),
         col    = c("steelblue", "tomato"),
         pch    = c(16, NA), lty = c(NA, 1), lwd = c(NA, 2),
         bty = "n", cex = 0.8)
  invisible(NULL)
}
