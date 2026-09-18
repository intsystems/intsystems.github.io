# frozen_string_literal: true

require 'pathname'

# The "edit on GitHub" button must open the file that actually rendered the page.
# `{% tf %}` falls back to the EN file when the current language has no copy
# (five root pages have no RU version, a few people have no bio in one language),
# so `_includes/edit.html` has to know which `_i18n/` files exist — and Liquid
# cannot look at the filesystem. Expose the list as `site.i18n_files`, a
# `{ "_i18n/ru/_people/ivanov_ii.md" => true }` hash for O(1) lookups by path.
Jekyll::Hooks.register :site, :after_init do |site|
  index = {}
  Dir.glob(File.join(site.source, '_i18n', '**', '*.md')).each do |path|
    index[Pathname.new(path).relative_path_from(Pathname.new(site.source)).to_s] = true
  end
  site.config['i18n_files'] = index
end
