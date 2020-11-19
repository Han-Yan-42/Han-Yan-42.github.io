---
title: "About"
permalink: /about/
header:
    image: "/images/shanghai1.jpeg"
---

A current student at University of Southern California majoring in Business Administration with concentrations in Data Science and Finance and Math Finance.

{% include base_path %}
{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
