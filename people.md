---
title: titles.people
permalink: /people/
meta_desc_en: "Lecturers at the Department of Intelligent Systems: Founders of the Department, Doctors of Science, PhD, teachers, Graduate Students, Instructors"
meta_desc_ru: "Преподаватели Кафедры интеллектуальных систем: Основатели кафедры, Доктора наук, Кандидаты наук, преподаватели, Аспиранты, семинаристы"
---

{% for role in site.global.people.roles %}

{% if role != 'template' %}

<div class="list-header">
  <h2 id="{% t site.global.people.roles.{{ role }} %}">{% t site.global.people.roles.{{ role }} %}</h2>
</div>

<div class="list people">
  {% for profile in site.people %}
    {% if profile.position == role %}
      <div class="list-item-people">
        <p class="list-post-title">
          {% if profile.avatar %}
            <a href="{{ site.baseurl }}{{ profile.url }}"><img class="profile-thumbnail" src="{{ site.baseurl_root }}/images/people/{{profile.avatar}}"></a>
          {% else %}
            <a href="{{ site.baseurl }}{{ profile.url }}"><img class="profile-thumbnail" src="{{ site.baseurl_root }}/images/people/default.jpg"></a>
          {% endif %}
          <a class="name" href="{{ site.baseurl }}{{ profile.url }}">{% t people.{{ profile.id | split: "/" | last }} %}</a>
        </p>
      </div>    
    {% endif %}
  {% endfor %}
</div>
{% endif %}

{% endfor %}
