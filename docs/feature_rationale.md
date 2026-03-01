# Feature Rationale

This document explains why each feature in `docs/feature_inventory.csv` exists.

## Data completeness and contact signal
- `addresses_present`: Places with any address data are generally better maintained.
- `has_addresses`: Same signal as above in model-specific featurizers; indicates basic listing completeness.
- `num_addresses`: Multiple addresses can indicate richer, actively managed records.
- `has_multiple_addresses`: Captures multi-location/address richness with a simple threshold.
- `address_completeness`: More complete address fields imply better record quality and potentially active businesses.
- `websites_present`: Presence of a website is a strong active-business proxy.
- `has_websites`: Same core website-presence signal for unified model features.
- `num_websites`: Multiple websites can indicate stronger digital presence.
- `has_multiple_websites`: Thresholded version of website count for robust non-linear effects.
- `phones_present`: Presence of a phone is a strong active-business proxy.
- `has_phones`: Same phone-presence signal for unified model features.
- `num_phones`: Multiple phones can indicate richer business metadata.
- `has_multiple_phones`: Thresholded phone-count signal.
- `socials_present`: Presence of socials is often associated with active places.
- `has_socials`: Same social-presence signal for unified model features.
- `num_socials`: Number of social links captures digital footprint depth.
- `has_multiple_socials`: Thresholded social-count signal.
- `contact_diversity`: More distinct contact channels usually means stronger listing quality.
- `has_full_contact_info`: Indicates complete contact surface (web+phone+social), often correlated with open status.
- `completeness_score`: Aggregate completeness proxy across key schema fields.
- `rich_profile`: Thresholded completeness score for robust rule-like behavior.

## Source coverage and provenance diversity
- `sources_n`: Number of sources supporting the record (rules baseline alias).
- `num_sources`: Same source-count signal in model featurizers.
- `has_multiple_sources`: Basic evidence corroboration signal.
- `has_many_sources`: Stronger corroboration threshold.
- `single_source`: Fragility signal; single-source records can be less reliable.
- `num_datasets`: Number of distinct source datasets; diversity can improve confidence in status.
- `has_multiple_datasets`: Thresholded dataset-diversity signal.
- `multi_dataset_with_contacts`: Interaction capturing corroboration plus direct contact richness.
- `multiple_sources_with_contacts`: Interaction capturing source breadth plus contact richness.

## Category, brand, and name semantics
- `has_primary_category`: Category metadata presence signal.
- `has_alternate_categories`: Richer category assignment signal.
- `num_categories`: Category breadth signal; can reflect listing detail and type complexity.
- `has_brand`: Brand presence can indicate structured, maintained records.
- `has_chain_pattern`: Name contains known chain tokens; chains can have distinct closure dynamics.
- `has_name`: Primary name availability signal.
- `name_length`: Simple lexical characteristic; very short/long names can reflect different listing patterns.
- `has_short_name`: Thresholded lexical pattern signal.
- `has_long_name`: Thresholded lexical pattern signal.
- `ohe_primary_category__*`: Explicit categorical identity signal for category-specific risk patterns.
- `category_closure_risk`: Fold-safe category-level closure prior to capture broad category risk differences.

## Temporal freshness and consistency
- `max_update_time`: Most recent source update timestamp; basis for recency features.
- `recency_days`: Core freshness signal measuring staleness vs dataset snapshot.
- `very_fresh`: Highly recent records often indicate active updates.
- `fresh`: Moderately recent update window.
- `stale`: Older records that may be more likely outdated or closed.
- `very_stale`: Very old records with stronger staleness risk.
- `source_temporal_diversity`: Variation in source update times; can indicate consistency or churn patterns.

## Geospatial context
- `geo_h3_cell_id`: Coarse location bucket to capture area-level closure patterns.
- `spatial_local_density`: Nearby-place density signal; local commercial intensity context.
- `neighbor_closed_rate`: Fold-safe local closure prior from nearby places.
- `same_category_neighbor_closed_rate`: Fold-safe local closure prior restricted to same category, capturing neighborhood+type effects.

## Interaction features
- `brand_with_contacts`: Brand signal strengthened when contact info exists.
- `recent_with_contacts`: Recency signal strengthened by contact richness.
- `single_source_no_socials`: Risk interaction for sparse provenance and weak digital footprint.

## Confidence-derived features (documented, excluded by policy)
- `overall_confidence`: Global provider confidence; excluded by team policy.
- `mean_source_conf`: Mean source confidence; excluded by team policy.
- `max_source_conf`: Max source confidence; excluded by team policy.
- `min_source_conf`: Min source confidence; excluded by team policy.
- `source_conf_std`: Source-confidence dispersion; excluded by team policy.
- `high_source_conf`: Thresholded source-confidence feature; excluded by team policy.
- `low_source_conf`: Thresholded source-confidence feature; excluded by team policy.
- `high_conf_with_contacts`: Confidence-contact interaction; excluded by team policy.
- `conf_very_high`: Global-confidence bucket; excluded by team policy.
- `conf_high`: Global-confidence bucket; excluded by team policy.
- `conf_medium`: Global-confidence bucket; excluded by team policy.
- `conf_low`: Global-confidence bucket; excluded by team policy.
- `conf_very_low`: Global-confidence bucket; excluded by team policy.

## Out-of-scope high-cost external features
- `url_live` (proposed, not in inventory): excluded for this phase because it requires external runtime website checks.
- `url_status_code` (proposed, not in inventory): excluded for this phase because it requires external runtime website checks.

## Feature aliases / overlap notes
- `sources_n` and `num_sources` intentionally overlap because rules and model pipelines currently use different names.
- `addresses_present` and `has_addresses`, `websites_present` and `has_websites`, `phones_present` and `has_phones`, `socials_present` and `has_socials` are semantically similar across different model scripts.
- `geo_cluster_id`: Tunable spatial clustering feature to capture local context with controlled dimensionality.
- `ohe_geo_cluster__*`: Explicit cluster identity signal without city-level one-hot explosion.
- `spatial_cluster_closed_rate`: Fold-safe cluster-level closure prior capturing local area risk patterns.
