from runner import experiment

def main():
    my_expirments = experiment("conf3.yaml")
    my_expirments.run()
    print("finised !")

if __name__ == '__main__':
    main()
