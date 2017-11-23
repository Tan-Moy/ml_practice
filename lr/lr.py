from numpy import *

def compute_error_for_given_points(b,m,points):
	totalError = 0
	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]
		totalError += (y-(m*x+b))**2
	return totalError/float(len(points))

def step_gradient(b_current,m_current,points,learningRate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]
		b_gradient += -(2/N) * (y-((m_current * x) + b_current))
		m_gradient += -(2/N) * x*(y-((m_current * x) + b_current))
	new_b = b_current - (learningRate * b_gradient)
	new_m = m_current - (learningRate * m_gradient)
	return[new_b,new_m]

def gradient_descent_runner(points,starting_b,starting_m,learning_rate,num_iterations):
	# y = mx + b
	b = starting_b
	m = starting_m

	for i in range(num_iterations):
		b,m = step_gradient(b,m,array(points),learning_rate)
	return[b,m]

def run():
	points = genfromtxt('data2.csv', delimiter=',')
	learning_rate = 0.01
	initial_b = 0
	initial_m = 0
	num_iterations = 1000
	[b,m] = gradient_descent_runner(points,initial_m,initial_b,learning_rate,num_iterations)
	print(b)
	print(m)
	error = compute_error_for_given_points(b,m,points)
	print(error)

if __name__ == '__main__':
	run()