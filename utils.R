library(tidyverse)

library(fs)
library(kilter)
library(rlang)


#' Compute average response grouping by Task, Participant, and stimulus_column, then by Task and stimulus_column only
#'
#' @param df Table with columns Task, Participant, {{stimulus_column}}, and {{response_column}}
#' @param stimulus_column Name of the stimulus column (e.g., Orientation or Numerosity)
#' @param response_column Name of the response column (e.g., OriResponse or NumerosityResponse)
#' @param resample logical, whether to sample data (not compatible with `predictions`)
#' @param predictions predicted values to replace original responses
#'
#' @returns table with columns Task, {{stimulus_column}}, and Response
#' @examples
#' 
#' # compute sample average
#' compute_average_response(results, Orientation, OriResponse)
#' 
#' # compute bootstrapped sample
#' compute_average_response(results, Orientation, OriResponse, resample = TRUE)
#' 
#' # compute predictions
#' compute_average_response(results, Orientation, OriResponse, preidictions = as.numeric(posterior_mu[1, ]))
compute_average_response <- function(df, stimulus_column, response_column, resample=FALSE, predictions=NULL){
  if (resample) df <- slice_sample(df, prop = 1, replace = TRUE)
  if (!is.null(predictions)) df <- df |> mutate({{ response_column }} := predictions)
  
  df |>
    group_by(Task, Participant, {{stimulus_column}}) |>
    summarise(Response = mean({{response_column}}), .groups = "drop") |>
    group_by(Task, {{stimulus_column}}) |>
    summarise(Response = mean(Response), .groups = "drop")
}

#' Bootstrap confidence interval and also compute sample average
#'
#' @param filename Filename to load from or to save computed results to
#' @param df Table with columns Task, Participant, {{stimulus_column}}, and {{response_column}}
#' @param stimulus_column Name of the stimulus column (e.g., Orientation or Numerosity)
#' @param response_column Name of the response column (e.g., OriResponse or NumerosityResponse)
#' @param CI confidence interval, defaults to `0.97`
#' @param R number of samples, defaults to `2000`
#' @param .progress logical, defaults to `TRUE`
#'
#' @returns Table with columns Task, {{stimulus_column}}, and Response, LowerCI, UpperCI
#'
#' @examples
#' bootstrap_group_averages(results, Orientation, OriResponse)
bootstrap_group_averages <- function(filename, df, stimulus_column, response_column, CI=0.97, R=2000, .progress = TRUE) {
  
  if (fs::file_exists(filename)) return(readRDS(filename))
  samples <- purrr::map(1:R, ~compute_average_response(df, {{stimulus_column}}, {{response_column}}, resample = TRUE), .progress = .progress)
  
  avgs <-
    samples |>
    list_rbind() |>
    group_by(Task, {{stimulus_column}}) |>
    summarise(LowerCI = kilter::lower_ci(Response, CI = CI),
              UpperCI = kilter::upper_ci(Response, CI = CI),
              .groups = "drop") |>
    right_join(compute_average_response(df, {{stimulus_column}}, {{response_column}}), by = c("Task",  as_string(ensym(stimulus_column))))
  
  saveRDS(avgs, filename)
  avgs
}

#' Credible interval from trial-level predictions
#'
#' @param filename Filename to load from or to save computed results to
#' @param df Table with columns Task, Participant, {{stimulus_column}}, and {{response_column}}
#' @param mu Matrix with posterior samples
#' @param stimulus_column Name of the stimulus column (e.g., Orientation or Numerosity)
#' @param response_column Name of the response column (e.g., OriResponse or NumerosityResponse)
#' @param CI confidence interval, defaults to `0.97`
#' @param .progress logical, defaults to `TRUE`
#'
#' @returns Table with columns Task, {{stimulus_column}}, and Response, LowerCI, UpperCI
#'
#' @examples
#' bootstrap_group_averages(results, Orientation, OriResponse)
posterior_group_averages_from_mu <- function(filename, df, mu, stimulus_column, response_column, CI=0.97, .progress = TRUE) {
  if (fs::file_exists(filename)) return(readRDS(filename))
  
  samples <- purrr::map(1:nrow(mu), ~compute_average_response(df, {{stimulus_column}}, {{response_column}}, predictions = mu[., ]), .progress = .progress)
  
  avgs <-
    samples |>
    list_rbind() |>
    group_by(Task, {{stimulus_column}}) |>
    summarise(LowerCI = kilter::lower_ci(Response, CI = CI),
              UpperCI = kilter::upper_ci(Response, CI = CI),
              Response = mean(Response),
              .groups = "drop")

  saveRDS(avgs, filename)
  avgs
}
