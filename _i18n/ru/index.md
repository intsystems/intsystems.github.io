<!-- Hero -->
<div class="hero">
  <div class="hero__bg"></div>
  <div class="hero__overlay"></div>
  <div class="hero__content">
    <h1 class="fade-in-left hero__title hero__title--sm">Кафедра интеллектуальных систем</h1>
    <p class="fade-in-right">
        Мы готовим специалистов в области прикладной математики и физики — от бакалавриата до аспирантуры. Наши исследования охватывают теорию машинного обучения, интеллектуальные системы и практические приложения. Основываясь на базе ВЦ РАН, мы объединяем академическое превосходство и сотрудничество с индустрией.
    </p>
    <div class="fade-in-right hero__socials">
        {% include social-links.html link_class="hero__social-link" skip_rutube=true %}
    </div>
  </div>
</div>

<!-- News Section -->
<section class="section-block">
    <h2>Новости</h2>
    <div class="news-scroll-container">
        <div class="news-track">
        {% assign news_sorted = site.posts | where: "lang", "ru" | sort: "date" | reverse %}
        {% if news_sorted.size > 0 %}
            {% for post in news_sorted limit:10 %}
            <a class="news-block" href="{{ site.baseurl }}{{ post.url }}">
                {% if post.important %}
                <div class="news-block__badge-row"><span class="news-important-badge">ВАЖНОЕ</span></div>
                {% endif %}
                <p class="news-title">{{ post.title }}</p>
                <p class="news-date">{{ post.date | date: "%d.%m.%Y" }}</p>
                <p class="news-excerpt">{{ post.excerpt }}</p>
            </a>
            {% endfor %}
        {% else %}
            Нет новостей
        {% endif %}
        </div>
    </div>
</section>

<!-- About Section -->
<section class="fade-in-section">
    <h2>О кафедре</h2>
    <div class="section-lead">
    <p>Кафедра интеллектуальных систем в Физтехе (МФТИ) является ведущим центром образования и исследований в области прикладной математики, науки о данных и искусственного интеллекта. Кафедра предлагает программы бакалавриата и магистратуры по направлению «Прикладная математика и физика» со специализациями в области науки о данных, проектирования интеллектуальных систем и машинного обучения.</p>
    <p>Кафедра была основана академиком Константином Владимировичем Рудаковым и развивалась в рамках научной школы академика Юрия Ивановича Журавлева. Она базируется в Вычислительном центре Российской академии наук. В нашем составе — известные профессора, молодые ученые и эксперты из индустрии, средний возраст которых составляет 35 лет.</p>
    <p>Исследования кафедры охватывают машинное обучение, многомерную статистику, глубокое обучение, выбор моделей, генеративные нейронные сети и анализ сложных данных. Прикладные проекты включают анализ текста и изображений, обработку биомедицинских сигналов и интерфейсы «мозг-компьютер». Кафедра активно сотрудничает с международными университетами, научными центрами и высокотехнологичными компаниями, предлагая студентам уникальные возможности для стажировок, двойных дипломов и совместных исследований.</p>
    <p>Мы ценим открытость, инновации и постоянное совершенствование, поддерживая студентов стипендиями и личным наставничеством. Присоединяйтесь к нам, чтобы учиться, исследовать и внедрять инновации в области интеллектуальных систем!</p>
    </div>
</section>

<!-- Department Statistics -->
<div class="stats">
    <div class="stats__grid">
        <div class="stat fade-in-left">
            <p class="stat__value">2003</p>
            <p class="stat__label">год основания кафедры</p>
        </div>
        <div class="stat fade-in-right">
            <p class="stat__value">>50%</p>
            <p class="stat__label">выпускников защитили кандидатские диссертации</p>
        </div>
        <div class="stat fade-in-left">
            <p class="stat__value"><35</p>
            <p class="stat__label">средний возраст преподавателей курсов</p>
        </div>
        <div class="stat fade-in-right">
            <p class="stat__value">170+</p>
            <p class="stat__label">open source проектов на <a href="https://github.com/{{ site.github }}">GitHub</a></p>
        </div>
        <div class="stat fade-in-left">
            <p class="stat__value stat__value--md">каждый<br>семестр</p>
            <p class="stat__label">студенты представляют <a href="{{ site.baseurl }}/materials/nir">научные отчеты</a>: paper-code-presentation</p>
        </div>
        <div class="stat fade-in-right">
            <p class="stat__value stat__value--sm">NeurIPS,<br>ICML, ICLR,<br>AISTATS</p>
            <p class="stat__label">top-tier конференции публикуют наши исследования</p>
        </div>
    </div>
</div>

<!-- Personalities Section -->
<section id="personalities" class="fade-in-section section-block">
    <h2>Личности</h2>
    <p class="section-lead">
        Мы гордимся нашими основателями и преподавателями, которые внесли значительный вклад в область машинного обучения. Их работа проложила путь к современным достижениям в области искусственного интеллекта.
    </p>
    <div class="list people">
    {% assign featured_people = "zhuravlyov_yv,rudakov_kv,vorontsov_kv,strijov_vv" | split: "," %}
    {% for person_id in featured_people %}
        {% include person-card.html id=person_id %}
    {% endfor %}
    </div>
</section>

<!-- Courses Section -->
<section class="fade-in-section">
    <h2>Курсы</h2>
    <p>
        Мы предлагаем широкий спектр курсов по прикладной математике, анализу данных и машинному обучению как для студентов бакалавриата, так и для магистрантов. Наша учебная программа разработана для обеспечения прочной теоретической базы наряду с практическими навыками, необходимыми в индустрии.
    </p>
    {% for type in site.global.course.types %}
        {% if type == 'bachelor' or type == 'master' %}
            <div class="list-header">
                <h3 id="{% t site.global.course.types.{{ type }} %}">{% t site.global.course.types.{{ type }} %}</h3>
            </div>
            <div class="list-course">
                {% for course in site.course %}
                    {% if course.type contains type %}
                    <a class="course-name" href="{{ site.baseurl }}{{ course.url }}">
                    <div class="list-item-course">
                        <p class="list-item-course-title">
                        {% t courses.{{ course.id | split: "/" | last }} %}
                        </p>
                    </div>
                    </a>
                    {% endif %}
                {% endfor %}
            </div>
        {% endif %}
    {% endfor %}
</section>

<!-- Full-width Image Before Research Section -->
<div class="fullwidth-figure">
<img loading="lazy" src="{{ site.baseurl_root }}/images/main/zhuravlev_rudakov_merged.jpg" alt="">
</div>

<!-- Research Section -->
<section class="fade-in-section">
    <h2>Научная работа</h2>
    <div class="research-section">
        <p>
            Мы открыто публикуем результаты исследований и приглашаем к сотрудничеству студентов, исследователей и промышленные компании.
        </p>
        <div class="research-grid">
            <div class="fade-in-left research-block">
                <a class="research-block-title" href="{{ site.baseurl }}/materials/nir">Научные исследования</a>
                <p class="research-block__text">
                    Наша кафедра проводит фундаментальные и прикладные исследования в области машинного обучения, анализа данных, искусственного интеллекта и смежных областей.
                    Результаты публикуются в открытом доступе и представляются на международных конференциях. Мы приветствуем совместные проекты и новые идеи!
                </p>
                <p class="research-block__footer">
                    <strong>Научные направления:</strong> распознавание образов, обработка естественного языка, биомедицинские сигналы, генеративные модели, теория машинного обучения
                </p>
            </div>
            <div class="fade-in-right research-block">
                <a class="research-block-title" href="{{ site.baseurl }}/materials/thesis">Дипломные работы</a>
                <p class="research-block__text">
                    Студенты участвуют в реальных исследованиях, готовят дипломные работы и публикуют свои результаты.
                    Мы поддерживаем открытое опубликование кода и статей и приглашаем всех к сотрудничеству по темам дипломных работ и исследовательским проектам.
                </p>
                <p class="research-block__footer">
                    <strong>Работа студентов:</strong> публикации, дипломные работы бакалавров и магистров, кандидатские диссертации
                </p>
            </div>
            <div class="fade-in-left research-block">
                <a class="research-block-title" href="{{ site.baseurl }}/materials/scholarship">Стипендии</a>
                <p class="research-block__text">
                    Мы поддерживаем исследования наших студентов, присуждая несколько стипендий каждый семестр.
                    <a href="{{ site.baseurl }}/materials/scholarship/" class="link-strong">Научная стипендия имени К.В. Рудакова</a> присуждается студентам бакалавриата и магистратуры за академические и исследовательские достижения.
                    <strong>Спонсор: Forecsys Group.</strong>
                </p>
            </div>
            <div class="fade-in-right research-block">
                <div class="research-block-title">Стажировки</div>
                <p class="research-block__text">
                    С самого начала кафедра активно сотрудничает с базовыми организациями группы компаний Forecsys и участвует в совместных проектах с ведущими технологическими компаниями.
                </p>
                <div class="tag-list">
                    <p>Форексис</p>
                    <p>Антиплагиат</p>
                    <p>Яндекс</p>
                    <p>СБЕР</p>
                </div>
            </div>
        </div>
    </div>
</section>

<!-- Image carousel -->
<section class="fade-in-section">
    <h2>Наша жизнь</h2>
    <p>
        Здесь мы делимся некоторыми моментами из жизни нашей кафедры: учебные мероприятия, защиты дипломов, встречи выпускников.
    </p>
    <div id="carousel-section">
        <div id="carousel-demo" class="carousel">
            <div class="carousel-item">
                <img loading="lazy" src="{{ site.baseurl_root }}/images/life/bachelors-2025.jpeg" alt="Бакалавриат 2025">
                <p>Бакалавриат, выпуск 2025</p>
            </div>
            <div class="carousel-item">
                <img loading="lazy" src="{{ site.baseurl_root }}/images/life/masters-2025.jpeg" alt="Магистратура 2025">
                <p>Магистратура, выпуск 2025</p>
            </div>
            <div class="carousel-item">
                <img loading="lazy" src="{{ site.baseurl_root }}/images/life/bachelors-2024.jpeg" alt="Бакалавриат 2024">
                <p>Бакалавриат, выпуск 2024</p>
            </div>
            <div class="carousel-item">
                <img loading="lazy" src="{{ site.baseurl_root }}/images/life/masters-2024.jpeg" alt="Магистратура 2024">
                <p>Магистратура, выпуск 2024</p>
            </div>
        </div>
    </div>
</section>
