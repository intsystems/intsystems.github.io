# frozen_string_literal: true

# Jekyll 4 memoizes the results of the `absolute_url` / `relative_url` filters
# in `site.filter_cache` for the lifetime of the Site object.
# jekyll-multiple-languages-plugin renders the site once per language on that
# same object (changing `baseurl` in between), so without this hook every
# RU page would inherit the EN URLs cached during the first pass — breaking
# canonical links and the RU sitemap.
Jekyll::Hooks.register :site, :after_reset do |site|
  site.filter_cache.clear
end
