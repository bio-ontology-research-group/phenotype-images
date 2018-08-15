import json
import csv

def find(number, filepath):
    num = number.split(".")
    num = int(num[0])
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)

        return data['response']['docs'][num]

def firstFilter(filepath):
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)

        filters = {}
        filters['sex'] = ['FEMALE', 'MALE', "UNKNOWN"]
        filters['imaging_method_label'] = []
        filters['mutation_type'] = []
        filters['taxon'] = []
        filters['phenotype_default_ontologies'] = []

        for p in data['response']['docs']:
            if 'imaging_method_label' in p and p['imaging_method_label'] not in filters['imaging_method_label']:
                filters['imaging_method_label'].append(p['imaging_method_label'])
            if 'mutation_type' in p and p['mutation_type'] not in filters['mutation_type']:
                filters['mutation_type'].append(p['mutation_type'])
            if 'taxon' in p and p['taxon'] not in filters['taxon']:
                filters['taxon'].append(p['taxon'])
            if 'phenotype_default_ontologies' in p and p['phenotype_default_ontologies'] not in filters['phenotype_default_ontologies']:
                filters['phenotype_default_ontologies'].append(p['phenotype_default_ontologies'])

        json_string = json.dumps(filters)

        return json.loads(json_string)


def Search(filepath, csvf, terms):
    terms = json.loads(terms)
    start = ['name', 'pca_x', 'pca_y']

    with open('/home/sara/PhpstormProjects/d3map/temp.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(start)
        with open(filepath) as json_file:
            data = json.load(json_file)

            with open(csvf, 'r') as csvfile:
                csvfile = csvfile.readlines()[1:]
                for line in csvfile:
                        split = line.split(',')
                        # image location in file
                        num = split[0].split('.')[0]
                        # get image json info
                        check = data['response']['docs'][int(num)]
                        # image bool value
                        image = False

                        val = search(terms, check)
                        if val:
                            writer.writerow([split[0], split[1], split[2]])


def search(terms, check):
    for i in terms:
        for tag in terms[i]:
            if i in check:
                if tag in check[i]:
                    return True

    return False
