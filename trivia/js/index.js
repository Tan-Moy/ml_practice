$(window).scroll(function () {
  $(".fade").css("opacity", 1 - $(window).scrollTop() / 300);
  console.log($(window).scrollTop())
});