var dataset = [[5, 20], [480, 90], [250, 50], [100, 33], [330, 95],
[410, 12], [475, 44], [25, 67], [85, 21], [220, 88]];

var w = 500;
var h = 100;

var svgContainer = d3.select('body')
						.append('svg')
						.attr('with',w)
						.attr('height',h);

var circlesContainer = svgContainer.selectAll('circle')
							.data(dataset)
							.enter()
							.append('circle')

circlesContainer.attr("cx", function(d) {
				return d[0];
				})
				.attr("cy", function(d) {
				return d[1];
				})
				.attr("r", function(d){
					return Math.sqrt(h-d[1]);
				});

var textContainer = svgContainer.selectAll('text').data(dataset).enter().append("text");

textContainer.text(d => d[0] + "," + d[1])
	.attr("x", d => d[0]).attr("y", d => d[1])
	.attr("font-family", "sans-serif")
	.attr("font-size", "11px")
	.attr("fill", "red");