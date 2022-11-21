from runner import experiment

def main():
    my_expirments = experiment("test.yaml")
    my_expirments.run()
    print("finised !")

if __name__ == '__main__':
    main()
