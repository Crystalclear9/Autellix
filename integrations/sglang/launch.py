def main():
    import sys
    from autellix.runtime.server import main as serve
    serve(["--backend", "sglang", *sys.argv[1:]])


if __name__ == "__main__":
    main()
