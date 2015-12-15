import os

def populate():
	a = [10,76,90,15,91,13,98,81,86,7,17,85,176,80,2,175,73,3,4,89,174,5,92,97,84,74,88,75,1,184,181,78,11,179,9,12,71,182,83,77,8,177,87,16,178,95,6,14,94,70,82,183,93,72,180,79,96]
	for i in range(len(a)):
		c = Category.objects.create(name=a[i])
		c.save()

# Start execution here!
if __name__ == '__main__':
    print "Starting Rango population script..."
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tangowithdjango.settings')
    from rango.models import Category
    populate()
