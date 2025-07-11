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
#' compute_average_absolute_response(results, Orientation, OriResponse)
#' 
#' # compute bootstrapped sample
#' compute_average_absolute_response(results, Orientation, OriResponse, resample = TRUE)
#' 
#' # compute predictions
#' compute_average_absolute_response(results, Orientation, OriResponse, predictions = as.numeric(posterior_mu[1, ]))
compute_average_absolute_response <- function(df, stimulus_column, response_column, resample=FALSE, predictions=NULL){
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
#' @param averaging_function Function used for averaging
#' @param CI confidence interval, defaults to `0.97`
#' @param R number of samples, defaults to `2000`
#' @param .progress logical, defaults to `TRUE`
#'
#' @returns Table with columns Task, {{stimulus_column}}, and Response, LowerCI, UpperCI
#'
#' @examples
#' bootstrap_group_averages(results, Orientation, OriResponse)
bootstrap_group_averages <- function(filename, df, stimulus_column, response_column, averaging_function, CI=0.97, R=2000, .progress = TRUE) {
  
  if (fs::file_exists(filename)) return(readRDS(filename))
  samples <- purrr::map(1:R, ~averaging_function(df, {{stimulus_column}}, {{response_column}}, resample = TRUE), .progress = .progress) |> list_rbind()
  
  grouping_columns <- setdiff(colnames(samples), "Response")

  avgs <-
    samples |>
    group_by(across(all_of(grouping_columns))) |>
    summarise(LowerCI = kilter::lower_ci(Response, CI = CI),
              UpperCI = kilter::upper_ci(Response, CI = CI),
              .groups = "drop") |>
    right_join(averaging_function(df, {{stimulus_column}}, {{response_column}}), by = grouping_columns)
  
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
posterior_group_averages_from_mu <- function(filename, df, mu, stimulus_column, response_column, averaging_function, CI=0.97, .progress = TRUE) {
  if (fs::file_exists(filename)) return(readRDS(filename))
  
  samples <- purrr::map(1:nrow(mu), ~averaging_function(df, {{stimulus_column}}, {{response_column}}, predictions = mu[., ]), .progress = .progress) |> list_rbind()
  
  grouping_columns <- setdiff(colnames(samples), "Response")
  
  avgs <-
    samples |>
    group_by(across(all_of(grouping_columns))) |>
    summarise(LowerCI = kilter::lower_ci(Response, CI = CI),
              UpperCI = kilter::upper_ci(Response, CI = CI),
              Response = mean(Response),
              .groups = "drop")

  saveRDS(avgs, filename)
  avgs
}



#' Comparing models via leave-one-out information criterion
#'
#' @param model_names list of model names
#' @param loos list of loos (for all models)
#'
#' @returns Table with columns model, elpd_diff, se_diff, weight (with order based on elpd_diff)
#'
#' @examples
#' summarize_loo_comparison(model_names, loos)
summarize_loo_comparison <- function(loos) {
  loo_table <- as_tibble(loo::loo_compare(loos), rownames = "model") |>
    dplyr::left_join(as_tibble(loo::loo_model_weights(loos), rownames = "model"), by = "model") |>
    mutate(dELPD = glue("{round(elpd_diff, 1)} ± {round(se_diff, 1)}"),
           weight = round(x, 3)) |>
    select(model, dELPD, weight)
  loo_table
}

#' Plotting model predictions per task
#'
#' @param model String with model name
#' @param posterior Table of posterior predictions grouped by task and stimulus with average responses, LowerCI and UpperCI 
#' @param bootstrapped Table bootstrapped behavioral averages grouped by task and stimulus with average responses, LowerCI and UpperCI 
#'
#' @returns ggplot: Posterior predictions per task 
#'
#' @examples
#' plot_model_predictions(a_model, posterior_mu, bootstrapped_ci, Orientation, OriResponse)
plot_model_predictions <- function(model, posterior, bootstrapped, stimulus_column, response_column) {
  ggplot(data = posterior, aes(x = {{stimulus_column}}, y = {{response_column}}, ymin = LowerCI, ymax = UpperCI, fill = Task)) +
    geom_ribbon(alpha = 0.7) +
    geom_line(aes(color = Task)) +
    geom_point(data = bootstrapped, aes(color = Task)) +
    geom_pointrange(data = bootstrapped, aes(color = Task)) +
    labs(title = model) +
    coord_equal()
}


#' Compute average relative response per Task and relative stimulus
#'
#' @param df Table with columns Task, Participant, {{stimulus_column}}, and {{response_column}}
#' @param stimulus_column Name of the stimulus column (e.g., Orientation or Numerosity)
#' @param response_column Name of the response column (e.g., OriResponse or NumerosityResponse)
#' @param resample logical, whether to sample data (not compatible with `predictions`)
#' @param predictions predicted values to replace original responses
#'
#' @returns Table with relative stimulus and average response per Task and Rel_Stimulus
#'
#' @examples
#' # compute relative sample average
#' compute_average_relative_response(results, Orientation, OriResponse)
#' 
#' # compute relative bootstrapped sample
#' compute_average_relative_response(results, Orientation, OriResponse, resample = TRUE)
#' 
#' # compute relative predictions
#' compute_average_relative_response(results, Orientation, OriResponse, predictions = as.numeric(posterior_mu[1, ]))
compute_average_relative_response <- function(df, stimulus_column, response_column, resample=FALSE, predictions=NULL) {
  if (!is.null(predictions)) df <- df |> mutate({{ response_column }} := predictions)
  df <- 
    df |> 
    group_by(Participant) |> 
    mutate(Rel_Stimulus = lag({{ stimulus_column }}) - {{ stimulus_column }},
           Rel_Response = {{ response_column }} - {{ stimulus_column }}) |>
    ungroup() |>
    
    # drop the first trial, as it has no prior orientation we can compute relative to
    filter(!IsFirstTrial)
  compute_average_absolute_response(df, Rel_Stimulus, Rel_Response, resample = resample)
}
