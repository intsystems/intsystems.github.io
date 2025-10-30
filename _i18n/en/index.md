<div style="position: relative; min-height: 60vh; display: flex; align-items: center; justify-content: center; text-align: center; margin: auto auto; width: 100vw; left: 50%; transform: translateX(-50%); overflow: hidden;">
  <div style="position: absolute; top: 0; bottom: 0; left: 0; right: 0; background: url('{{ site.baseurl_root }}/images/main/вц_ран.jpg') center/cover no-repeat; filter: blur(5px); transform: scale(1.1); z-index: 0;"></div>
  <div style="position: absolute; top: 0; bottom: 0; left: 0; right: 0; background-color: #000; z-index: 1; opacity: 0.6"></div>
  <div class="hero-content" style="position: relative; z-index: 2; color: #fff; width: 90%; max-width: 1200px; padding: 1.5rem;">
    <p class="fade-in-left" style="font-size: clamp(0.9rem, 3vw, 1rem); margin-bottom: 0;">Department of</p>
    <h1 class="fade-in-left" style="color: #fff; margin-bottom: 1.5rem; font-size: clamp(1.8rem, 8vw, 3rem); font-weight: 600; line-height: 1.2; margin-top: 0;">Intelligent Systems</h1>
    <p class="fade-in-right">
        We train specialists in applied mathematics and physics — from bachelor's to PhD. Our research spans machine learning theory, intelligent systems, and real-world applications. Based at the Dorodnicyn Computing Centre of RAS, we unite academic excellence with industry collaboration.
    </p>
    <div class="fade-in-right" style="display: flex; justify-content: center; align-items: center; gap: clamp(0.8rem, 4vw, 1.5rem); flex-wrap: wrap;">
    {% if site.github %}
    <a href="https://github.com/{{site.github}}" target="_blank" style="font-size: clamp(1.2rem, 5vw, 1.5rem); padding: 0.5rem; min-width: 44px; min-height: 44px; display: inline-flex; align-items: center; justify-content: center; text-decoration: none;"><i class="fa fa-github"></i></a>
    {% endif %}
    {% if site.mail %}
    <a href="#" onclick="copyEmailToClipboard(event, '{{site.mail}}')" style="font-size: clamp(1.2rem, 5vw, 1.5rem); padding: 0.5rem; min-width: 44px; min-height: 44px; display: inline-flex; align-items: center; justify-content: center; text-decoration: none;"><i class="fa fa-envelope-o"></i></a>
    {% endif %}
    {% if site.telegram %}
    <a href="{{site.telegram}}"  target="_blank" style="font-size: clamp(1.2rem, 5vw, 1.5rem); padding: 0.5rem; min-width: 44px; min-height: 44px; display: inline-flex; align-items: center; justify-content: center; text-decoration: none;"><i class="fa fa-telegram"></i></a>
    {% endif %}
    {% if site.youtube %}
    <a href="{{site.youtube}}"  target="_blank" style="font-size: clamp(1.2rem, 5vw, 1.5rem); padding: 0.5rem; min-width: 44px; min-height: 44px; display: inline-flex; align-items: center; justify-content: center; text-decoration: none;"><i class="fa fa-youtube-play"></i></a>
    {% endif %}
    </div>
    </div>
</div>

<!-- News Section -->
<section style="margin: 2rem 0;">
    <h2>News</h2>
    <div style="display: flex; align-items: center; position: relative;">
        <div class="news-scroll-container" style="overflow-x: auto; flex: 1;">
            <div style="display: flex; gap: 1.5rem; min-width: 0; padding: 1rem 0.5rem;">
            {% if site.posts and site.posts.size > 0 %}
                {% assign news_sorted = site.posts | where: "lang", "en" | sort: 'date' | reverse %}
                {% for post in news_sorted limit:10 %}
                    <a class="news-block" href="{{ site.baseurl }}{{ post.url }}">
                        <div style="display: flex; justify-content: flex-end; align-items: flex-start;">
                            {% if post.important %}
                                <span class="news-important-badge">IMPORTANT</span>
                            {% endif %}
                        </div>
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
    </div>
</section>

<!-- About Section -->
<section class="fade-in-section">
    <h2>About</h2>
    <div style="margin-top: 1rem;">
    <p>The Department of Intelligent Systems at the Phystech School of Applied Mathematics and Informatics (MIPT) is a leading center for education and research in applied mathematics, data science, and artificial intelligence. The department offers bachelor’s and master’s programs in “Applied Mathematics and Physics” with specializations in Data Science, Intelligent Systems Design, and Machine Learning.</p>
    <p>Founded by academician Konstantin Vladimirovich Rudakov and developed within the scientific school of academician Yuri Ivanovich Zhuravlev, the department is based at the Computing Center of the Russian Academy of Sciences. Our faculty includes renowned professors, young scientists, and industry experts, with an average age of 35 years.</p>
    <p>Research at the department covers machine learning, multivariate statistics, deep learning, model selection, generative neural networks, and analysis of complex data. Applied projects include text and image analysis, biomedical signal processing, and brain-computer interfaces. The department actively collaborates with international universities, research centers, and high-tech companies, offering students unique opportunities for internships, double degrees, and joint research.</p>
    <p>We value openness, innovation, and continuous improvement, supporting students with scholarships and personal mentorship. Join us to study, research, and innovate in the field of intelligent systems!</p>
    </div>
</section>

<!-- Department Statistics -->
<div style="margin: 3rem auto 2.5rem auto; width: 100%;">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr)); gap: clamp(1.5rem, 3vw, 2rem) clamp(2rem, 4vw, 4rem); text-align: center; padding: 1.5rem 0;">
        <div class="fade-in-left" style="display: flex; flex-direction: column; align-items: center;">
            <p style="font-size: 4rem; font-weight: 600; line-height: 0.9; margin-bottom: 0.5rem;">2003</p>
            <p style="max-width: 220px;font-size: 1rem;">year the department was founded</p>
        </div>
        <div class="fade-in-right" style="display: flex; flex-direction: column; align-items: center;">
            <p style="font-size: 4rem; font-weight: 600; line-height: 0.9; margin-bottom: 0.5rem;">>50%</p>
            <p style="max-width: 220px;font-size: 1rem;">of graduates defended PhD dissertations</p>
        </div>
        <div class="fade-in-left" style="display: flex; flex-direction: column; align-items: center;">
            <p style="font-size: 4rem; font-weight: 600; line-height: 0.9; margin-bottom: 0.5rem;"><35</p>
            <p style="max-width: 220px; font-size: 1rem;">years average age of courses instructors</p>
        </div>
        <div class="fade-in-right" style="display: flex; flex-direction: column; align-items: center;">
            <p style="font-size: 4rem; font-weight: 600; line-height: 0.9; margin-bottom: 0.5rem;">170+</p>
            <p style="max-width: 220px; font-size: 1rem;">open source projects on <a href="https://github.com/{{site.github}}" >GitHub</a></p>
        </div>
        <div class="fade-in-left" style="display: flex; flex-direction: column; align-items: center;">
            <p style="font-size: 2.8rem; font-weight: 600; line-height: 0.9; margin-bottom: 0.5rem;">every<br>semester</p>
            <p style="max-width: 220px; font-size: 1rem;">students submit <a href="{{ site.baseurl }}/materials/nir">research reports</a>: paper-code-presentation</p>
        </div>
        <div class="fade-in-right" style="display: flex; flex-direction: column; align-items: center;">
            <p style="font-size: 2.2rem; font-weight: 600; line-height: 1.1; margin-bottom: 0.5rem;">NeurIPS,<br>ICML, ICLR,<br>AISTATS</p>
            <p style="max-width: 220px; font-size: 1rem;">top-tier conferences publish our research</p>
        </div>
    </div>
</div>

<!-- Personalities Section -->
<section id="personalities" class="fade-in-section" style="margin: 2rem 0;">
  <h2>Personalities</h2>
    <p style="margin-top: 1rem;">
        We are proud of our founders and lecturers, who have made significant contributions to the field of intelligent systems. Their work has paved the way for advancements in artificial intelligence and machine learning.
    </p>
    {% assign featured_people = "zhuravlyov_yv,rudakov_kv,vorontsov_kv,strijov_vv" | split: "," %}
    <div class="list people">
    {% for person_id in featured_people %}
        {% for profile in site.people %}
            {% assign profile_id = profile.id | split: "/" | last %}
            {% if profile_id == person_id %}
                <div class="list-item-people">
                    <p class="list-post-title">
                    {% if profile.avatar %}
                        <a href="{{ site.baseurl }}{{ profile.url }}"><img class="profile-thumbnail" src="{{ site.baseurl_root }}/images/people/{{profile.avatar}}"></a>
                    {% else %}
                        <a href="{{ site.baseurl }}{{ profile.url }}"><img class="profile-thumbnail" src="{{ site.baseurl_root }}/images/people/default.jpg"></a>
                    {% endif %}
                    <a class="name" href="{{ site.baseurl }}{{ profile.url }}">{% t people.{{ profile_id }} %}</a>
                    </p>
                </div>
            {% endif %}
        {% endfor %}
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
<div style="width: 100vw; position: relative; left: 50%; transform: translateX(-50%); margin: 3.5rem 0;">
<img src="{{ site.baseurl_root }}/images/main/zhuravlev_rudakov_merged.jpg" style="width: 100vw; max-width: 100%; height: auto; object-fit: cover;  border-radius: 0px">
</div>

<!-- Research Section -->
<section class="fade-in-section">
    <h2>Research</h2>
    <div style="margin: 0 0 3rem 0; max-width: 1200px;">
        <p>
            We openly publish research results and invite collaboration with students, researchers, and industry partners.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(360px, 100%), 1fr)); gap: 2.5rem; margin: 2.5rem 0;">
            <div class="fade-in-left research-block" >
                <a class="research-block-title" href="{{ site.baseurl }}/materials/nir">Scientific Research</a>
                <p style="margin: 1rem 0;">
                    Our department conducts fundamental and applied research in machine learning, data analysis, artificial intelligence, and related fields.
                    Results are published in open access and presented at international conferences. We welcome joint projects and new ideas!
                </p>
                <p style="padding-top: 1rem; border-top: 1px solid #eee; text-align: left;">
                    <strong>Focus areas:</strong> ML algorithms, AI research, data science
                </p>
            </div>
            <div class="fade-in-right research-block" >
                <a class="research-block-title" href="{{ site.baseurl }}/materials/thesis">Theses</a>
                <p style="margin: 1rem 0;">
                    Students participate in real research, prepare diploma theses, and publish their results.
                    We support open publication of code and articles, and invite everyone to collaborate on thesis topics and research projects.
                </p>
                <p style="padding-top: 1rem; border-top: 1px solid #eee; text-align: left;">
                    <strong>Student work:</strong> Bachelor's & Master's theses, publications
                </p>
            </div>
            <div class="fade-in-left research-block" >
                <a class="research-block-title" href="{{ site.baseurl }}/materials/scholarship">Scholarships</a>
                <p style="margin: 1rem 0;">
                    We support the research of our students by awarding several scholarships each semester.
                    The <a href="{{ site.baseurl }}/materials/scholarship/" style=" font-weight: 600; text-decoration: none;">scientific academic scholarship named after K.V. Rudakov</a> is awarded to undergraduate and graduate students for academic and research excellence.
                    <strong>Sponsored by Forecsys Group.</strong>
                </p>
            </div>
            <div class="fade-in-right research-block" >
                <div class="research-block-title">Internships</div>
                <p style="margin: 1rem 0;">
                    Since the beginning, the department has been actively cooperating with the base companies of the Forecsys Group of Companies and participates in joint projects with leading tech companies.
                </p>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                    <p style="background: #f0f0f0; padding: 0.3rem 0.8rem; border-radius: 1rem;">Forecsys</p>
                    <p style="background: #f0f0f0; padding: 0.3rem 0.8rem; border-radius: 1rem;">Antiplagiat</p>
                    <p style="background: #f0f0f0; padding: 0.3rem 0.8rem; border-radius: 1rem;">Yandex</p>
                    <p style="background: #f0f0f0; padding: 0.3rem 0.8rem; border-radius: 1rem;">SBER</p>
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
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma-carousel@4.0.3/dist/css/bulma-carousel.min.css">
        <div id="carousel-demo" class="carousel">
            <div class="carousel-item">
                <img src="{{ site.baseurl_root }}/images/life/bachelors-2025.jpeg" alt="Bachelors 2025">
                <p>Bachelors 2025'</p>
            </div>
            <div class="carousel-item">
                <img src="{{ site.baseurl_root }}/images/life/masters-2025.jpeg" alt="Masters 2025">
                <p>Masters 2025'</p>
            </div>
            <div class="carousel-item">
                <img src="{{ site.baseurl_root }}/images/life/bachelors-2024.jpeg" alt="Bachelors 2024">
                <p>Bachelors 2024'</p>
            </div>
            <div class="carousel-item">
                <img src="{{ site.baseurl_root }}/images/life/masters-2024.jpeg" alt="Masters 2024">
                <p>Masters 2024'</p>
            </div>
        </div>
    </div>
</section>
