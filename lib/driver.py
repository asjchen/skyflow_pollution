import input_util

def main():	
	parser = input_util.make_top_parser()
	args = parser.parse_args()
	args.func(args)

if __name__ == '__main__':
	main()