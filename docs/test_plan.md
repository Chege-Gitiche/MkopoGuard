# Test plan

How MkopoGuard's requirements are proved. The strategy behind it is in `docs/test_strategy.md`.
**Version:** 1.0, 2026-10-19.

This file is checked by a script: every test named in the matrix must exist, and every
requirement must have a test or a planned gap. Run `python scripts/check_traceability.py`.

## 1. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | An officer can sign in with email and password | Must |
| FR-02 | A failed sign-in gives the same message whatever was wrong, so it never reveals who has an account | Must |
| FR-03 | An officer can log out, which clears the session | Must |
| FR-04 | Every officer page and API route needs a valid sign-in; a missing or expired one leads back to the login page | Must |
| FR-05 | Accounts are created and deactivated only by an administrator script (no public sign-up); one account per email; deactivating ends sessions at once | Must |
| FR-06 | Roles are officer, admin and demo; the demo account can score applicants but nothing it does is saved | Must |
| FR-07 | An officer can score an applicant from their answers alone (thin file) | Must |
| FR-08 | An officer can attach an M-Pesa statement (CSV); the features it gives match the ones the model was trained on | Must |
| FR-09 | Invalid or contradictory answers are refused, naming the field | Must |
| FR-10 | An invalid statement is refused, naming the row or column: missing columns, bad values, edited balances, wrong period, too short, over 1 MB, not a CSV | Must |
| FR-11 | The result shows the probability of default, the risk band, the decision and the threshold for the loan term | Must |
| FR-12 | The result explains itself: 3 reasons that raised the risk, 2 that lowered it, in plain English, and how much everything else added | Must |
| FR-13 | An applicant without a statement is never given statement-based reasons | Must |
| FR-14 | Unusual but valid inputs are scored with a warning | Should |
| FR-15 | Each assessment is saved with its answers, reasons, model version and threshold, but never the raw transactions | Must |
| FR-16 | An officer sees their history newest first, a page at a time | Must |
| FR-17 | History can be filtered by band, decision, reference prefix and Kenyan date range | Should |
| FR-18 | An officer can open one saved assessment in full | Must |
| FR-19 | Model info shows the model version, test score, decision thresholds and bands | Should |
| FR-20 | The assessment form can be filled from ten synthetic sample applicants | Should |
| FR-21 | The browser checks the form with the same rules as the API, which stays the real check | Should |
| FR-22 | "Not for real lending decisions" appears on every page, with every decision, in the API docs and in the model card | Must |
| FR-23 | The whole journey works in a real browser: sign in, assess a sample, see the result, find it in history, log out | Must |
| FR-24 | The API is documented at `/docs` with a working example | Should |
| FR-25 | A public health check names the model and reports the database | Should |

## 2. Non-functional requirements

| ID | Requirement | Priority |
|---|---|---|
| NFR-01 | Data isolation: an officer never sees another officer's assessments; someone else's record looks exactly like a missing one (404) | Must |
| NFR-02 | Passwords are stored only as salted bcrypt hashes, and weak passwords are refused | Must |
| NFR-03 | Sign-in is rate-limited: 5 failures per address and email in 15 minutes, 20 attempts per address in 5 minutes, then 429 with Retry-After | Must |
| NFR-04 | Login tokens are short-lived and tamper-proof: expired, edited, unsigned or wrongly signed tokens are refused | Must |
| NFR-05 | The session cookie is HttpOnly, SameSite=Strict and HTTPS-only on the live site; changes made with it need the app's own header (cross-site request forgery guard) | Must |
| NFR-06 | Every response has security headers and a strict Content-Security-Policy; responses with personal data are never cached; pages never turn data into HTML | Must |
| NFR-07 | No password, token or email in logs, errors or validation messages; an unexpected error is a plain 500 with a reference | Must |
| NFR-08 | The app refuses to start with unsafe settings: no or weak secret key, SQLite in production, an unmigrated database, a corrupted model bundle | Must |
| NFR-09 | Model quality floor: test PR-AUC 0.35 or more, calibration error (ECE) 0.03 or less | Must |
| NFR-10 | The served model is exactly the one tested: checksummed files, golden examples replay exactly, reloading changes nothing | Must |
| NFR-11 | The model moves risk the designed way for every input it uses, except three documented flips | Should |
| NFR-12 | Gender, country and age group are never model inputs; the fairness audit's findings are recorded and published | Must |
| NFR-13 | No data leakage: the test split is locked, transactions stay inside their window, generated data stays plausible | Must |
| NFR-14 | Accessible: Lighthouse accessibility 90 or more on every screen, every field labelled, colour never the only signal, visible keyboard focus | Must |
| NFR-15 | Usable from 360 px wide, and reflows without sideways scrolling at 320 px (400% zoom) | Should |
| NFR-16 | `/predict` p95 under 500 ms locally and under 1.5 s on the live free tier, at 10 concurrent users, with no errors | Should |
| NFR-17 | Database migrations go up and down cleanly and keep existing history | Must |
| NFR-18 | Times are stored in UTC and shown in Kenyan time | Should |

## 3. Traceability matrix

Each requirement and the automated tests that prove it. A "Gap" names the step that will add
the missing test.

| Requirement | Tests | Status |
|---|---|---|
| FR-01 | `tests/unit/test_auth.py::test_login_returns_a_short_lived_bearer_token`<br>`tests/unit/test_cookie_auth.py::test_app_login_sets_an_httponly_strict_cookie_and_returns_no_token`<br>`tests/unit/test_cookie_auth.py::test_the_cookie_logs_the_officer_in`<br>`tests/unit/test_auth.py::test_login_ignores_email_case_and_spaces` | Covered |
| FR-02 | `tests/unit/test_auth.py::test_every_failed_login_gets_the_same_answer`<br>`tests/unit/test_auth.py::test_an_unknown_email_still_checks_a_password`<br>`tests/unit/test_hardening.py::test_being_rate_limited_does_not_reveal_whether_an_account_exists` | Covered |
| FR-03 | `tests/unit/test_cookie_auth.py::test_logout_clears_the_session_cookie`<br>`tests/unit/test_cookie_auth.py::test_logout_works_with_no_session_or_an_expired_one` | Covered |
| FR-04 | `tests/unit/test_auth.py::test_me_needs_a_valid_token`<br>`tests/unit/test_auth.py::test_an_expired_token_is_refused_by_the_api`<br>`tests/unit/test_predict.py::test_predict_needs_a_login`<br>`tests/unit/test_history.py::test_history_needs_a_login`<br>`tests/unit/test_openapi.py::test_every_protected_endpoint_says_it_needs_a_login_and_documents_401`<br>`tests/unit/test_frontend.py::test_app_pages_check_the_session_first` | Covered |
| FR-05 | `tests/unit/test_auth.py::test_the_script_creates_and_deactivates_accounts`<br>`tests/unit/test_auth.py::test_one_account_per_email`<br>`tests/unit/test_auth.py::test_deactivating_an_account_ends_its_session_at_once` | Covered |
| FR-06 | `tests/unit/test_auth.py::test_roles_are_enforced`<br>`tests/unit/test_predict.py::test_the_demo_account_gets_a_result_but_nothing_is_saved`<br>`tests/unit/test_history.py::test_the_demo_account_has_an_empty_history` | Covered |
| FR-07 | `tests/unit/test_predict.py::test_every_sample_scores_as_recorded_through_the_api`<br>`tests/unit/test_samples.py::test_sample_scores_as_recorded` | Covered |
| FR-08 | `tests/unit/test_statement.py::test_a_clean_statement_is_accepted`<br>`tests/unit/test_statement.py::test_a_statement_gives_every_statement_feature`<br>`tests/data/test_statement_real.py::test_csv_upload_reproduces_the_training_features_and_score` | Covered |
| FR-09 | `tests/unit/test_schemas.py::test_out_of_range_or_unknown_values_are_refused`<br>`tests/unit/test_schemas.py::test_contradicting_answers_are_refused`<br>`tests/unit/test_schemas.py::test_the_reference_cannot_be_a_phone_id_number_or_name`<br>`tests/unit/test_schemas.py::test_the_application_date_cannot_be_in_the_future`<br>`tests/unit/test_predict.py::test_a_bad_profile_field_is_named` | Covered |
| FR-10 | `tests/unit/test_statement.py::test_missing_extra_and_repeated_columns_are_all_named`<br>`tests/unit/test_statement.py::test_a_bad_value_is_reported_with_its_row_number`<br>`tests/unit/test_statement.py::test_an_edited_balance_is_caught`<br>`tests/unit/test_statement.py::test_a_statement_that_is_too_short_is_refused`<br>`tests/unit/test_predict.py::test_a_statement_for_another_period_is_refused`<br>`tests/unit/test_predict.py::test_a_huge_upload_is_refused_without_reading_it_all`<br>`tests/unit/test_predict.py::test_a_file_that_is_not_a_csv` | Covered |
| FR-11 | `tests/unit/test_predict.py::test_the_decision_follows_the_threshold_for_the_term`<br>`tests/unit/test_schemas.py::test_bands`<br>`tests/unit/test_decision.py::test_longer_loans_can_carry_more_risk` | Covered |
| FR-12 | `tests/unit/test_reasons.py::test_always_exactly_three_raising_and_two_lowering`<br>`tests/unit/test_reasons.py::test_every_reason_is_a_readable_sentence`<br>`tests/unit/test_reasons.py::test_units_and_other_factors_add_up_to_the_score`<br>`tests/data/test_reasons_real.py::test_three_plus_two_different_reasons_for_everyone` | Covered |
| FR-13 | `tests/unit/test_reasons.py::test_thin_file_applicants_never_get_statement_reasons`<br>`tests/unit/test_predict.py::test_a_thin_file_applicant_is_never_given_statement_reasons`<br>`tests/data/test_reasons_real.py::test_thin_file_applicants_get_no_statement_reasons` | Covered |
| FR-14 | `tests/unit/test_predict.py::test_unusual_inputs_are_scored_with_a_warning` | Covered |
| FR-15 | `tests/unit/test_predict.py::test_an_officers_prediction_is_saved_with_its_model_version`<br>`tests/unit/test_predict.py::test_saved_inputs_keep_features_but_never_the_raw_transactions`<br>`tests/unit/test_predict.py::test_nothing_is_saved_when_the_input_is_bad` | Covered |
| FR-16 | `tests/unit/test_history.py::test_newest_first`<br>`tests/unit/test_history.py::test_pages_cover_everything_once`<br>`tests/unit/test_history.py::test_a_page_past_the_end_is_empty_not_an_error` | Covered |
| FR-17 | `tests/unit/test_history.py::test_filter_by_band_and_decision`<br>`tests/unit/test_history.py::test_filter_by_reference_prefix_ignores_case_and_wildcards`<br>`tests/unit/test_history.py::test_filter_by_kenyan_calendar_days`<br>`tests/unit/test_history.py::test_date_from_after_date_to_is_refused`<br>`tests/unit/test_frontend.py::test_history_filters_match_the_api_query_parameters` | Covered |
| FR-18 | `tests/unit/test_history.py::test_a_prediction_made_through_the_api_comes_back_in_full`<br>`tests/unit/test_frontend.py::test_detail_page_shows_every_answer` | Covered |
| FR-19 | `tests/unit/test_predict.py::test_model_info_describes_the_model_and_its_rule`<br>`tests/unit/test_predict.py::test_model_info_needs_a_login` | Covered |
| FR-20 | `tests/unit/test_frontend.py::test_samples_are_served_for_load_sample`<br>`tests/unit/test_frontend.py::test_samples_offered_are_the_committed_samples`<br>`tests/unit/test_samples.py::test_there_are_samples_for_every_band_and_thin_file` | Covered |
| FR-21 | `tests/unit/test_frontend.py::test_browser_checks_match_the_api_rules`<br>`tests/unit/test_frontend.py::test_assessment_form_asks_exactly_what_the_api_needs`<br>`tests/unit/test_frontend.py::test_assessment_choices_use_the_values_the_api_accepts` | Covered |
| FR-22 | `tests/unit/test_frontend.py::test_every_page_carries_the_disclaimer_in_its_footer`<br>`tests/unit/test_frontend.py::test_the_decision_carries_the_disclaimer_with_it`<br>`tests/unit/test_openapi.py::test_the_docs_warn_it_is_not_for_real_lending`<br>`tests/unit/test_model_card.py::test_card_says_it_is_not_for_real_lending_at_the_top` | Covered |
| FR-23 | | Gap: Playwright journey test, step 7.4 (checked by hand in steps 6.4 to 6.6) |
| FR-24 | `tests/unit/test_openapi.py::test_the_docs_example_scores_in_the_api`<br>`tests/unit/test_openapi.py::test_every_endpoint_has_a_summary_one_group_and_a_description` | Covered |
| FR-25 | `tests/unit/test_api_app.py::test_health_is_up_and_names_the_model`<br>`tests/unit/test_api_app.py::test_health_needs_no_login`<br>`tests/unit/test_database.py::test_health_reports_the_database` | Covered |
| NFR-01 | `tests/unit/test_history.py::test_officer_a_gets_404_on_officer_bs_prediction`<br>`tests/unit/test_history.py::test_someone_elses_prediction_looks_exactly_like_a_missing_one`<br>`tests/unit/test_history.py::test_the_list_never_includes_anyone_elses_predictions`<br>`tests/unit/test_predict.py::test_each_officers_predictions_carry_their_own_id` | Covered |
| NFR-02 | `tests/unit/test_auth.py::test_a_password_is_stored_only_as_a_salted_hash`<br>`tests/unit/test_auth.py::test_weak_passwords_are_refused`<br>`tests/unit/test_database.py::test_repr_never_shows_the_password_hash` | Covered |
| NFR-03 | `tests/unit/test_hardening.py::test_five_failures_lock_that_ip_and_email_pair_for_15_minutes`<br>`tests/unit/test_hardening.py::test_one_ip_trying_many_emails_is_capped`<br>`tests/unit/test_hardening.py::test_the_sixth_wrong_password_is_refused_even_if_the_seventh_is_right`<br>`tests/unit/test_hardening.py::test_an_attacker_elsewhere_cannot_lock_the_real_officer_out` | Covered |
| NFR-04 | `tests/unit/test_auth.py::test_an_expired_token_is_refused`<br>`tests/unit/test_auth.py::test_a_tampered_token_is_refused`<br>`tests/unit/test_auth.py::test_an_unsigned_token_is_refused`<br>`tests/unit/test_auth.py::test_a_token_signed_with_another_key_is_refused`<br>`tests/unit/test_cookie_auth.py::test_a_tampered_cookie_is_401`<br>`tests/unit/test_api_app.py::test_token_lifetime_must_be_short` | Covered |
| NFR-05 | `tests/unit/test_cookie_auth.py::test_app_login_sets_an_httponly_strict_cookie_and_returns_no_token`<br>`tests/unit/test_cookie_auth.py::test_the_cookie_is_https_only_on_the_live_site`<br>`tests/unit/test_cookie_auth.py::test_a_change_with_the_cookie_but_without_the_app_header_is_refused` | Covered |
| NFR-06 | `tests/unit/test_hardening.py::test_every_response_has_a_request_id_and_security_headers`<br>`tests/unit/test_hardening.py::test_api_responses_with_personal_data_are_never_cached`<br>`tests/unit/test_cookie_auth.py::test_pages_and_api_get_the_strict_policy`<br>`tests/unit/test_frontend.py::test_no_inline_scripts_or_event_handlers`<br>`tests/unit/test_frontend.py::test_scripts_never_write_html` | Covered |
| NFR-07 | `tests/unit/test_hardening.py::test_logs_never_contain_the_password_the_token_or_the_email`<br>`tests/unit/test_hardening.py::test_a_validation_error_never_echoes_what_was_sent`<br>`tests/unit/test_hardening.py::test_an_unexpected_error_is_a_plain_500_with_a_reference`<br>`tests/unit/test_api_app.py::test_health_reveals_nothing_secret` | Covered |
| NFR-08 | `tests/unit/test_api_app.py::test_secret_key_is_required`<br>`tests/unit/test_api_app.py::test_short_or_placeholder_secret_is_refused`<br>`tests/unit/test_api_app.py::test_production_refuses_a_local_sqlite_database`<br>`tests/unit/test_database.py::test_app_refuses_to_start_on_an_unmigrated_database`<br>`tests/unit/test_api_app.py::test_app_refuses_to_start_with_a_corrupted_bundle` | Covered |
| NFR-09 | `tests/unit/test_quality_gates.py::test_the_model_clears_the_pr_auc_floor`<br>`tests/unit/test_quality_gates.py::test_the_model_is_well_calibrated`<br>`tests/unit/test_champion_bundle.py::test_bundle_matches_the_final_test_report` | Covered |
| NFR-10 | `tests/unit/test_champion_bundle.py::test_bundle_loads_and_replays_its_golden_examples`<br>`tests/unit/test_bundle.py::test_an_edited_file_is_refused`<br>`tests/unit/test_bundle.py::test_reloaded_bundle_predicts_exactly_like_the_original`<br>`tests/unit/test_bundle.py::test_changed_golden_answer_is_caught_even_with_a_matching_checksum` | Covered |
| NFR-11 | `tests/unit/test_behaviour_champion.py::test_champion_moves_risk_the_expected_way`<br>`tests/data/test_behaviour_real.py::test_every_check_except_the_known_flips_passes`<br>`tests/data/test_behaviour_real.py::test_the_known_flips_still_fail` | Covered |
| NFR-12 | `tests/unit/test_champion_bundle.py::test_no_forbidden_column_is_a_model_input`<br>`tests/unit/test_fairness.py::test_audited_attributes_are_never_model_inputs`<br>`tests/data/test_fairness_real.py::test_recorded_flags_still_hold_for_the_shipped_model`<br>`tests/unit/test_model_card.py::test_card_reports_every_fairness_flag` | Covered |
| NFR-13 | `tests/unit/test_split.py::test_test_set_is_locked_by_default`<br>`tests/data/test_split_real.py::test_committed_split_has_not_been_changed_by_hand`<br>`tests/unit/test_features.py::test_transactions_on_or_after_the_application_date_are_ignored`<br>`tests/unit/test_transactions.py::test_all_transactions_are_inside_the_statement_window`<br>`tests/data/test_checks_real.py::test_generated_data_is_plausible` | Covered |
| NFR-14 | `tests/unit/test_frontend.py::test_every_form_control_has_a_label`<br>`tests/unit/test_frontend.py::test_one_h1_and_no_skipped_heading_levels`<br>`tests/unit/test_frontend.py::test_band_colours_pass_contrast`<br>`tests/unit/test_frontend.py::test_every_band_tag_names_its_band_in_text`<br>`tests/unit/test_frontend.py::test_focus_is_always_visible` | Covered (Lighthouse by hand: `docs/accessibility.md`; axe automated in step 7.4) |
| NFR-15 | `tests/unit/test_frontend.py::test_grids_never_need_more_room_than_the_screen` | Covered |
| NFR-16 | | Gap: Locust load test of `/predict`, step 7.4 |
| NFR-17 | `tests/unit/test_database.py::test_migrations_go_down_and_up_again`<br>`tests/unit/test_database.py::test_migrations_keep_existing_history`<br>`tests/unit/test_database.py::test_models_and_migrations_agree` | Covered |
| NFR-18 | `tests/unit/test_database.py::test_times_are_always_returned_in_utc`<br>`tests/unit/test_history.py::test_filter_by_kenyan_calendar_days`<br>`tests/unit/test_statement.py::test_times_with_a_timezone_are_converted_to_kenyan_time` | Covered |

## 4. Test cases

Written in step 7.3: ID, requirement, steps, test data and expected result for the happy paths,
boundaries and negative cases.
