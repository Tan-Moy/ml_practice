var dataset = [[5, 20], [480, 90], [250, 50], [100, 33], [330, 95],
[410, 12], [475, 44], [25, 67], [85, 21], [220, 88]];

var w = 500;
var h = 200;
var p = 20;

var svgContainer = d3.select('body')
						.append('svg')
						.attr('with',w)
						.attr('height',h);

var circlesContainer = svgContainer.selectAll('circle')
							.data(dataset)
							.enter()
							.append('circle')

var textContainer = svgContainer.selectAll('text').data(dataset).enter().append("text");


var scaleConfigX = d3.scaleLinear()
.domain([0,d3.max(dataset,function(d){return d[0]})])
.range([p, w - p]);

var scaleConfigY = d3.scaleLinear()
.domain([0, d3.max(dataset, function(d) { return d[1]; })])
.range([h - p, p]);


circlesContainer.attr("cx", function(d) {
				return scaleConfigX(d[0]);
				})
				.attr("cy", function(d) {
				return scaleConfigY(d[1]);
				})
				.attr("r", function(d){
					return Math.sqrt(h-d[1]);
				});

textContainer.text(d => d[0] + "," + d[1])
	.attr("x", d => scaleConfigX(d[0]))
	.attr("y", d => scaleConfigY(d[1]))
	.attr("font-family", "sans-serif")
	.attr("font-size", "11px")
	.attr("fill", "red");

var xAxis = d3.axisBottom(scaleConfigX);

svgContainer.append('g')
.attr("class", "axis")
.attr("transform", "translate(0," + (h - p) + ")")
.call(xAxis)

var yAxis = d3.axisLeft(scaleConfigY).ticks(5);

svgContainer.append('g')
.attr("class", "axis")
.attr("transform", "translate(" + p + ",0)")
.call(yAxis)
