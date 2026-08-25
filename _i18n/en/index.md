<!-- Hero -->
<div class="hero">
  <div class="hero__bg"></div>
  <div class="hero__overlay"></div>
  <div class="hero__content">
    <p class="fade-in-left hero__eyebrow">Department of</p>
    <h1 class="fade-in-left hero__title">Intelligent Systems</h1>
    <p class="fade-in-right">
        We train specialists in applied mathematics and physics — from bachelor's to PhD. Our research spans machine learning theory, intelligent systems, and real-world applications. Based at the Dorodnicyn Computing Centre of RAS, we unite academic excellence with industry collaboration.
    </p>
    <div class="fade-in-right hero__socials">
        {% include social-links.html link_class="hero__social-link" skip_rutube=true %}
    </div>
  </div>
</div>

<!-- News Section -->
<section class="section-block">
    <h2>News</h2>
    <div class="news-scroll-container">
        <div class="news-track">
        {% assign news_sorted = site.posts | where: "lang", "en" | sort: "date" | reverse %}
        {% if news_sorted.size > 0 %}
            {% for post in news_sorted limit:10 %}
            <a class="news-block" href="{{ site.baseurl }}{{ post.url }}">
                {% if post.important %}
                <div class="news-block__badge-row"><span class="news-important-badge">IMPORTANT</span></div>
                {% endif %}
                <p class="news-title">{{ post.title }}</p>
                <p class="news-date">{{ post.date | date: "%d.%m.%Y" }}</p>
                <p class="news-excerpt">{{ post.excerpt }}</p>
            </a>
            {% endfor %}
        {% else %}
            No news available
        {% endif %}
        </div>
    </div>
</section>

<!-- About Section -->
<section class="fade-in-section">
    <h2>About</h2>
    <div class="section-lead">
    <p>The Department of Intelligent Systems at the Phystech School of Applied Mathematics and Informatics (MIPT) is a leading center for education and research in applied mathematics, data science, and artificial intelligence. The department offers bachelor’s and master’s programs in “Applied Mathematics and Physics” with specializations in Data Science, Intelligent Systems Design, and Machine Learning.</p>
    <p>Founded by academician Konstantin Vladimirovich Rudakov and developed within the scientific school of academician Yuri Ivanovich Zhuravlev, the department is based at the Computing Center of the Russian Academy of Sciences. Our faculty includes renowned professors, young scientists, and industry experts, with an average age of 35 years.</p>
    <p>Research at the department covers machine learning, multivariate statistics, deep learning, model selection, generative neural networks, and analysis of complex data. Applied projects include text and image analysis, biomedical signal processing, and brain-computer interfaces. The department actively collaborates with international universities, research centers, and high-tech companies, offering students unique opportunities for internships, double degrees, and joint research.</p>
    <p>We value openness, innovation, and continuous improvement, supporting students with scholarships and personal mentorship. Join us to study, research, and innovate in the field of intelligent systems!</p>
    </div>
</section>

<!-- Department Statistics -->
<div class="stats">
    <div class="stats__grid">
        <div class="stat fade-in-left">
            <p class="stat__value">2003</p>
            <p class="stat__label">year the department was founded</p>
        </div>
        <div class="stat fade-in-right">
            <p class="stat__value">>50%</p>
            <p class="stat__label">of graduates defended PhD dissertations</p>
        </div>
        <div class="stat fade-in-left">
            <p class="stat__value"><35</p>
            <p class="stat__label">years average age of courses instructors</p>
        </div>
        <div class="stat fade-in-right">
            <p class="stat__value">170+</p>
            <p class="stat__label">open source projects on <a href="https://github.com/{{ site.github }}">GitHub</a></p>
        </div>
        <div class="stat fade-in-left">
            <p class="stat__value stat__value--md">every<br>semester</p>
            <p class="stat__label">students submit <a href="{{ site.baseurl }}/materials/nir">research reports</a>: paper-code-presentation</p>
        </div>
        <div class="stat fade-in-right">
            <p class="stat__value stat__value--sm">NeurIPS,<br>ICML, ICLR,<br>AISTATS</p>
            <p class="stat__label">top-tier conferences publish our research</p>
        </div>
    </div>
</div>

<!-- Personalities Section -->
<section id="personalities" class="fade-in-section section-block">
    <h2>Personalities</h2>
    <p class="section-lead">
        We are proud of our founders and lecturers, who have made significant contributions to the field of intelligent systems. Their work has paved the way for advancements in artificial intelligence and machine learning.
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
    <h2>Courses</h2>
    <p>
        We offer a range of courses in applied mathematics, data science, and machine learning for both bachelor's and master's students. Our curriculum is designed to provide a strong theoretical foundation along with practical skills needed in the industry.
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
    <h2>Research</h2>
    <div class="research-section">
        <p>
            We openly publish research results and invite collaboration with students, researchers, and industry partners.
        </p>
        <div class="research-grid">
            <div class="fade-in-left research-block">
                <a class="research-block-title" href="{{ site.baseurl }}/materials/nir">Scientific Research</a>
                <p class="research-block__text">
                    Our department conducts fundamental and applied research in machine learning, data analysis, artificial intelligence, and related fields.
                    Results are published in open access and presented at international conferences. We welcome joint projects and new ideas!
                </p>
                <p class="research-block__footer">
                    <strong>Focus areas:</strong> ML algorithms, AI research, data science
                </p>
            </div>
            <div class="fade-in-right research-block">
                <a class="research-block-title" href="{{ site.baseurl }}/materials/thesis">Theses</a>
                <p class="research-block__text">
                    Students participate in real research, prepare diploma theses, and publish their results.
                    We support open publication of code and articles, and invite everyone to collaborate on thesis topics and research projects.
                </p>
                <p class="research-block__footer">
                    <strong>Student work:</strong> Bachelor's & Master's theses, publications
                </p>
            </div>
            <div class="fade-in-left research-block">
                <a class="research-block-title" href="{{ site.baseurl }}/materials/scholarship">Scholarships</a>
                <p class="research-block__text">
                    We support the research of our students by awarding several scholarships each semester.
                    The <a href="{{ site.baseurl }}/materials/scholarship/" class="link-strong">scientific academic scholarship named after K.V. Rudakov</a> is awarded to undergraduate and graduate students for academic and research excellence.
                    <strong>Sponsored by Forecsys Group.</strong>
                </p>
            </div>
            <div class="fade-in-right research-block">
                <div class="research-block-title">Internships</div>
                <p class="research-block__text">
                    Since the beginning, the department has been actively cooperating with the base companies of the Forecsys Group of Companies and participates in joint projects with leading tech companies.
                </p>
                <div class="tag-list">
                    <p>Forecsys</p>
                    <p>Antiplagiat</p>
                    <p>Yandex</p>
                    <p>SBER</p>
                </div>
            </div>
        </div>
    </div>
</section>

<!-- Image carousel -->
<section class="fade-in-section">
    <h2>Our Life</h2>
    <p>
        Here we share some moments from our department's life, including events, student activities, and memorable experiences.
    </p>
    <div id="carousel-section">
        <div id="carousel-demo" class="carousel">
            <div class="carousel-item">
                <img loading="lazy" src="{{ site.baseurl_root }}/images/life/bachelors-2025.jpeg" alt="Bachelors 2025">
                <p>Bachelors 2025'</p>
            </div>
            <div class="carousel-item">
                <img loading="lazy" src="{{ site.baseurl_root }}/images/life/masters-2025.jpeg" alt="Masters 2025">
                <p>Masters 2025'</p>
            </div>
            <div class="carousel-item">
                <img loading="lazy" src="{{ site.baseurl_root }}/images/life/bachelors-2024.jpeg" alt="Bachelors 2024">
                <p>Bachelors 2024'</p>
            </div>
            <div class="carousel-item">
                <img loading="lazy" src="{{ site.baseurl_root }}/images/life/masters-2024.jpeg" alt="Masters 2024">
                <p>Masters 2024'</p>
            </div>
        </div>
    </div>
</section>
