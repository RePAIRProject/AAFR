from runner import experiment

def main():
    my_expirments = experiment("conf2.yaml")
    my_expirments.run()
    print("finised !")

if __name__ == '__main__':
    main()
