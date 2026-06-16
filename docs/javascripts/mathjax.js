(function () {
  var currentScript = document.currentScript;
  var mathjaxSrc = new URL(
    "mathjax/tex-mml-chtml.min.js",
    currentScript ? currentScript.src : document.baseURI
  ).href;
  var loading = null;

  window.MathJax = {
    loader: {
      load: ["[tex]/boldsymbol"]
    },
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true,
      packages: { "[+]": ["boldsymbol"] }
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    },
    startup: {
      typeset: false
    }
  };

  function hasMath() {
    return !!document.querySelector(".arithmatex");
  }

  function loadMathJax() {
    if (window.MathJax && window.MathJax.typesetPromise) {
      return Promise.resolve();
    }
    if (loading) {
      return loading;
    }

    loading = new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src = mathjaxSrc;
      script.async = true;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
    return loading;
  }

  function renderMath() {
    if (!hasMath()) {
      return;
    }

    loadMathJax()
      .then(function () {
        if (window.MathJax.startup && window.MathJax.startup.promise) {
          return window.MathJax.startup.promise;
        }
        return Promise.resolve();
      })
      .then(function () {
        if (window.MathJax.typesetPromise) {
          return window.MathJax.typesetPromise([document.body]);
        }
        return Promise.resolve();
      })
      .catch(function (error) {
        console.warn("MathJax failed to load or render", error);
      });
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(renderMath);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMath);
  } else {
    renderMath();
  }
})();
