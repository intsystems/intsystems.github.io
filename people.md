---
title: titles.people
permalink: /people/
meta_desc_en: "Lecturers at the Department of Intelligent Systems: Founders of the Department, Doctors of Science, PhD, teachers, Graduate Students, Instructors"
meta_desc_ru: "Преподаватели Кафедры интеллектуальных систем: Основатели кафедры, Доктора наук, Кандидаты наук, преподаватели, Аспиранты, семинаристы"
---

<h1 class="sr-only">{% t titles.people %}</h1>

{% for role in site.global.people.roles %}

{% if role != 'template' %}

<div class="list-header">
  <h2 id="{% t site.global.people.roles.{{ role }} %}">{% t site.global.people.roles.{{ role }} %}</h2>
</div>

<div class="list people">
  {% assign group = site.people | where: "position", role %}
  {% for profile in group %}
    {% include person-card.html profile=profile %}
  {% endfor %}
</div>
{% endif %}

{% endfor %}
