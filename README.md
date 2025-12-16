# Intelligent Systems Department Website

[![License](https://badgen.net/github/license/intsystems/intsystems.github.io)](https://github.com/intsystems/intsystems.github.io/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/intsystems/intsystems.github.io)](https://github.com/intsystems/intsystems.github.io/graphs/contributors)

This repository contains the source code for the Intelligent Systems Department website. It is designed to provide information about our department, courses, lecturers, research, publications and contact details.

If you want to make a large contribution, please do it in a pull request, and ask [@kisnikser](https://github.com/kisnikser) for a review. Else if your changes are small (typos, small content edits), you can directly edit the files in GitHub web interface as follows.

## Blog

1. Create or edit a file in the [`_blogs`](_blogs/) folder. Use the existing files as a reference for the structure and required fields.
2. If your blog contains some images, add them to the [`images/blog/{blog-title}`](images/blog/) folder, where `blog-title` equals your `.md` filename under [`_blogs`](_blogs/). Please, compress the images to optimize loading times.

## Research Report

1. Go to the file [`_i18n/en/nir.md`](_i18n/en/nir.md).
2. If you add new semester, create a new section, e.g., `### 2025 Fall`, and create a table for your course (see previous semesters for examples).
3. Add your report information in the format:
   ```markdown
   | Student      | Topic      | Advisor               | Links                                    |
   | :----------- | :--------- | :-------------------- | :--------------------------------------- |
   | Name Surname | Topic name | Name Surname, PhD/DSc | [Paper](URL), [Code](URL), [Slides](URL) |
   ```

## Thesis

1. Go to the file [`_i18n/en/thesis.md`](_i18n/en/thesis.md).
2. If you add new year, create a new section, e.g., `### 2025`, and create a table for your course (see previous years for examples).
3. Add your report information in the format:
   ```markdown
   | Student      | Topic       | Advisor               | Link to Project | Link to Paper | Link to Slides |
   | :----------- | :---------- | :-------------------- | :-------------- | ------------- | -------------- |
   | Name Surname | Thesis name | Name Surname, PhD/DSc | [Project](URL)  | [Thesis](URL) | [Slides](URL)  |
   ```

## People

1. Create or edit a file in the [`_people`](_people/) folder. Use the existing files as a reference for the structure and required fields. Each file should be named using the format `lastname_firstname.md`.
2. Add a brief biography and any relevant links (e.g., personal website, scholar, elibrary) to the files in [`_i18n/en/_people/`](_i18n/en/_people/) and [`_i18n/ru/_people/`](_i18n/ru/_people/) directories. Name files consistently across languages, following the same `lastname_firstname` format.
3. Add the person's profile picture to the [`images/people/`](images/people/) folder. Please, compress the image to optimize loading times. Add image filename to the file from 1st step.
4. Update meta information `people` in the [`_i18n/en.yml`](_i18n/en.yml) and [`_i18n/ru.yml`](_i18n/ru.yml) files to include the new person, following the same `lastname_firstname` format.

## Courses

1. Create or edit a file in the [`_course`](_course/) folder. Use the existing files as a reference for the structure and required fields. Each file should be named using the format `course_name.md`.
2. Add course details and any relevant links (e.g., syllabus, resources) to the files in [`_i18n/en/_course/`](_i18n/en/_course/) and [`_i18n/ru/_course/`](_i18n/ru/_course/) directories. Name files consistently across languages, following the same `course_name` format.
3. Update meta information `courses` in the [`_i18n/en.yml`](_i18n/en.yml) and [`_i18n/ru.yml`](_i18n/ru.yml) files to include the new course, following the same `course_name` format.
